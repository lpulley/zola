use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use errors::Result;
use exif::Exif;

use crate::format::Format;
use crate::processor::ImageSource;
use crate::ResizeOperation;
use libs::image::DynamicImage;

/// Apply image rotation based on EXIF data
/// Returns `None` if no transformation is needed
pub fn fix_orientation(img: &DynamicImage, exif: &Exif) -> Option<DynamicImage> {
    let orientation =
        exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?.value.get_uint(0)?;
    match orientation {
        // Values are taken from the page 30 of
        // https://www.cipa.jp/std/documents/e/DC-008-2012_E.pdf
        // For more details check http://sylvana.net/jpegcrop/exif_orientation.html
        1 => None,
        2 => Some(img.fliph()),
        3 => Some(img.rotate180()),
        4 => Some(img.flipv()),
        5 => Some(img.fliph().rotate270()),
        6 => Some(img.rotate90()),
        7 => Some(img.fliph().rotate90()),
        8 => Some(img.rotate270()),
        _ => None,
    }
}

pub fn get_processed_filename(
    input: &ImageSource,
    op: &ResizeOperation,
    format: &Format,
) -> Result<String> {
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    op.hash(&mut hasher);
    format.hash(&mut hasher);
    let hash = hasher.finish();

    Ok(format!("{:016x}.{}", hash, format.extension()))
}
