use std::fmt;
use std::fs;
use std::fs::File;
use std::hash::Hasher;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;

use config::Config;
use errors::{anyhow, Context, Result};
use exif::Exif;
use libs::ahash::{HashMap, HashSet};
use libs::image::codecs::avif::AvifEncoder;
use libs::image::codecs::jpeg::JpegEncoder;
use libs::image::imageops::FilterType;
use libs::image::{DynamicImage, GenericImageView};
use libs::image::{EncodableLayout, ExtendedColorType, ImageEncoder, ImageFormat};
use libs::rayon::prelude::*;
use libs::sha2::Digest;
use libs::sha2::Sha256;
use libs::{image, webp};
use serde::{Deserialize, Serialize};
use utils::fs as ufs;

use crate::format::Format;
use crate::helpers::get_processed_filename;
use crate::{fix_orientation, ImageMeta, ResizeInstructions, ResizeOperation};

pub const RESIZED_SUBDIR: &str = "processed_images";

pub struct ImageSource {
    image: DynamicImage,
    meta: ImageMeta,
    exif: Option<Exif>,
    checksum: [u8; 32],
}

impl ImageSource {
    fn read(path: &Path) -> Result<Self> {
        let image = image::open(&path)?;
        let meta = ImageMeta::read(&path)
            .with_context(|| format!("Failed to read image: {}", path.display()))?;
        let exif = exif::Reader::new()
            .read_from_container(&mut std::io::BufReader::new(std::fs::File::open(&path)?))
            .ok();
        let checksum: [u8; 32] = Sha256::digest(fs::read(&path)?).into();

        Ok(ImageSource { image, meta, exif, checksum })
    }
}

impl std::cmp::PartialEq for ImageSource {
    fn eq(&self, other: &ImageSource) -> bool {
        // If the file checksum is equal, everything else must be equal.
        self.checksum == other.checksum
    }
}

impl std::cmp::Eq for ImageSource {}

impl std::hash::Hash for ImageSource {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // A hash of the file checksum is (for our purposes) as unique as a hash of the file itself.
        self.checksum.hash(state)
    }
}

impl fmt::Debug for ImageSource {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        f.debug_struct("ImageSource")
            .field("image", &self.image)
            .field("meta", &self.meta)
            .field(
                "exif",
                match &self.exif {
                    Some(exif) => &"HAS EXIF (FIXME)",
                    None => &"NO EXIF (FIXME)",
                },
            )
            .field("checksum", &self.checksum)
            .finish()
    }
}

#[derive(Debug)]
pub struct ImageSourceCache {
    cache_mutex: Mutex<HashMap<PathBuf, Arc<ImageSource>>>,
}

impl ImageSourceCache {
    pub fn new() -> Self {
        ImageSourceCache { cache_mutex: Mutex::new(HashMap::default()) }
    }

    pub fn read(&mut self, path: &Path) -> Result<Arc<ImageSource>> {
        // Take the cache mutex
        let mut cache = self
            .cache_mutex
            .lock()
            .map_err(|lock_err| anyhow!("Failed to get lock: {}", lock_err.to_string()))?;
        if cache.contains_key(path) {
            // Return a clone of the cached source Arc
            Ok(cache[path].clone())
        } else {
            // Read the source for the first time into a new Arc; clone it into the cache and return it
            let new_source = Arc::new(ImageSource::read(path)?);
            cache.insert(path.to_path_buf(), new_source.clone());
            Ok(new_source)
        }
    }
}

/// Holds all data needed to perform a resize operation
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageOp {
    source: Arc<ImageSource>,
    output_path: PathBuf,
    instr: ResizeInstructions,
    format: Format,
    /// Whether we actually want to perform that op.
    /// In practice we set it to true if the output file already
    /// exists and is not stale. We do need to keep the ImageOp around for pruning though.
    ignore: bool,
}

impl ImageOp {
    fn perform(&self) -> Result<()> {
        if self.ignore {
            return Ok(());
        }

        let mut img = match &self.source.exif {
            Some(exif) => {
                fix_orientation(&self.source.image, &exif).unwrap_or(self.source.image.clone())
            }
            None => self.source.image.clone(),
        };

        let img = match self.instr.crop_instruction {
            Some((x, y, w, h)) => img.crop(x, y, w, h),
            None => img,
        };
        let img = match self.instr.resize_instruction {
            Some((w, h)) => img.resize_exact(w, h, FilterType::Lanczos3),
            None => img,
        };

        let f = File::create(&self.output_path)?;
        let mut buffered_f = BufWriter::new(f);

        match self.format {
            Format::Png => {
                img.write_to(&mut buffered_f, ImageFormat::Png)?;
            }
            Format::Jpeg { quality } => {
                let mut encoder = JpegEncoder::new_with_quality(&mut buffered_f, quality);
                encoder.encode_image(&img)?;
            }
            Format::WebP { quality } => {
                let encoder = webp::Encoder::from_image(&img)
                    .map_err(|_| anyhow!("Unable to load this kind of image with webp"))?;
                let memory = match quality {
                    Some(q) => encoder.encode(q as f32),
                    None => encoder.encode_lossless(),
                };
                buffered_f.write_all(memory.as_bytes())?;
            }
            Format::Avif { quality, speed } => {
                let mut avif: Vec<u8> = Vec::new();
                let color_type = match img.color() {
                    image::ColorType::L8 => Ok(ExtendedColorType::L8),
                    image::ColorType::La8 => Ok(ExtendedColorType::La8),
                    image::ColorType::Rgb8 => Ok(ExtendedColorType::Rgb8),
                    image::ColorType::Rgba8 => Ok(ExtendedColorType::Rgba8),
                    image::ColorType::L16 => Ok(ExtendedColorType::L16),
                    image::ColorType::La16 => Ok(ExtendedColorType::La16),
                    image::ColorType::Rgb16 => Ok(ExtendedColorType::Rgb16),
                    image::ColorType::Rgba16 => Ok(ExtendedColorType::Rgba16),
                    image::ColorType::Rgb32F => Ok(ExtendedColorType::Rgb32F),
                    image::ColorType::Rgba32F => Ok(ExtendedColorType::Rgba32F),
                    c => Err(anyhow!("Unknown image color type '{:?}' for AVIF", c)),
                }?;
                let encoder = AvifEncoder::new_with_speed_quality(&mut avif, speed, quality);
                encoder.write_image(
                    &img.as_bytes(),
                    img.dimensions().0,
                    img.dimensions().1,
                    color_type,
                )?;
                buffered_f.write_all(&avif.as_bytes())?;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnqueueResponse {
    /// The final URL for that asset
    pub url: String,
    /// The path to the static asset generated
    pub static_path: String,
    /// New image width
    pub width: u32,
    /// New image height
    pub height: u32,
    /// Original image width
    pub orig_width: u32,
    /// Original image height
    pub orig_height: u32,
}

impl EnqueueResponse {
    fn new(
        url: String,
        static_path: PathBuf,
        meta: &ImageMeta,
        instr: &ResizeInstructions,
    ) -> Self {
        let static_path = static_path.to_string_lossy().into_owned();
        let (width, height) = instr.resize_instruction.unwrap_or(meta.size);
        let (orig_width, orig_height) = meta.size;

        Self { url, static_path, width, height, orig_width, orig_height }
    }
}

/// A struct into which image operations can be enqueued and then performed.
/// All output is written in a subdirectory in `static_path`,
/// taking care of file stale status based on timestamps
#[derive(Debug)]
pub struct Processor {
    base_url: String,
    output_dir: PathBuf,
    img_ops: HashSet<ImageOp>,
    source_cache: ImageSourceCache,
}

impl Processor {
    pub fn new(base_path: PathBuf, config: &Config) -> Processor {
        Processor {
            output_dir: base_path.join("static").join(RESIZED_SUBDIR),
            base_url: config.make_permalink(RESIZED_SUBDIR),
            img_ops: HashSet::default(),
            source_cache: ImageSourceCache::new(),
        }
    }

    pub fn set_base_url(&mut self, config: &Config) {
        self.base_url = config.make_permalink(RESIZED_SUBDIR);
    }

    pub fn num_img_ops(&self) -> usize {
        self.img_ops.len()
    }

    pub fn enqueue(
        &mut self,
        op: ResizeOperation,
        input_path: PathBuf,
        format: &str,
        quality: Option<u8>,
        speed: Option<u8>,
    ) -> Result<EnqueueResponse> {
        let source = self.source_cache.read(&input_path)?;

        // We get the output format
        let format = Format::from_args(source.meta.is_lossy(), format, quality, speed)?;
        // Now we have all the data we need to generate the output filename and the response
        let filename = get_processed_filename(&source, &op, &format)?;
        let url = format!("{}{}", self.base_url, filename);
        let static_path = Path::new("static").join(RESIZED_SUBDIR).join(&filename);
        let output_path = self.output_dir.join(&filename);
        let instr = ResizeInstructions::new(op, source.meta.size);
        let enqueue_response = EnqueueResponse::new(url, static_path, &source.meta, &instr);
        let img_op = ImageOp { ignore: output_path.exists(), source, output_path, instr, format };
        self.img_ops.insert(img_op);

        Ok(enqueue_response)
    }

    /// Run the enqueued image operations
    pub fn do_process(&mut self) -> Result<()> {
        if !self.img_ops.is_empty() {
            ufs::create_directory(&self.output_dir)?;
        }

        self.img_ops
            .par_iter()
            .map(|op| {
                op.perform().with_context(|| {
                    format!("Failed to process image: {}", op.input_path.display())
                })
            })
            .collect::<Result<()>>()
    }

    /// Remove stale processed images in the output directory
    pub fn prune(&self) -> Result<()> {
        // Do not create folders if they don't exist
        if !self.output_dir.exists() {
            return Ok(());
        }

        ufs::create_directory(&self.output_dir)?;
        let output_paths: HashSet<_> = self
            .img_ops
            .iter()
            .map(|o| o.output_path.file_name().unwrap().to_string_lossy())
            .collect();

        for entry in fs::read_dir(&self.output_dir)? {
            let entry_path = entry?.path();
            if entry_path.is_file() {
                let filename = entry_path.file_name().unwrap().to_string_lossy();
                if !output_paths.contains(&filename) {
                    fs::remove_file(&entry_path)?;
                }
            }
        }
        Ok(())
    }
}
