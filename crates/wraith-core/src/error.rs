//! Error types shared across the wraith pipeline.

use std::path::PathBuf;

/// Every failure mode of the PNG → SVG pipeline.
#[derive(Debug, thiserror::Error)]
pub enum WraithError {
    /// The input bytes are not a decodable PNG image.
    #[error("not a decodable PNG: {0}")]
    Decode(#[from] image::ImageError),

    /// The tracer rejected the prepared image.
    #[error("tracing failed: {0}")]
    Trace(String),

    /// The intermediate SVG could not be parsed by the minifier.
    #[error("intermediate SVG failed to parse: {0}")]
    SvgParse(String),

    /// A minifier job aborted.
    #[error("minification failed: {0}")]
    Minify(String),

    /// The minified SVG could not be serialized.
    #[error("minified SVG failed to serialize: {0:?}")]
    SvgWrite(oxvg_ast::error::XmlWriterError),

    /// An input path is not a file or directory.
    #[error("`{}` is not a file or directory", path.display())]
    BadInput {
        /// The offending path.
        path: PathBuf,
    },

    /// Output location is invalid for the given set of inputs.
    #[error("{0}")]
    BadOutput(String),
}

/// Convenience alias for results in this crate.
pub type Result<T, E = WraithError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_error_displays_prefix() {
        let err = WraithError::Decode(image::load_from_memory(b"junk").unwrap_err());
        assert!(err.to_string().starts_with("not a decodable PNG:"));
    }

    #[test]
    fn trace_error_includes_reason() {
        let err = WraithError::Trace(String::from("boom"));
        assert_eq!(err.to_string(), "tracing failed: boom");
    }

    #[test]
    fn parse_write_and_output_errors_render() {
        assert_eq!(
            WraithError::SvgParse(String::from("bad xml")).to_string(),
            "intermediate SVG failed to parse: bad xml"
        );
        assert_eq!(
            WraithError::Minify(String::from("job")).to_string(),
            "minification failed: job"
        );
        let utf8 = String::from_utf8(vec![0xFF]).expect_err("invalid utf8");
        let err = WraithError::SvgWrite(oxvg_ast::error::XmlWriterError::UTF8(utf8));
        assert!(
            err.to_string()
                .starts_with("minified SVG failed to serialize: UTF8(")
        );
        assert_eq!(
            WraithError::BadInput {
                path: PathBuf::from("nope")
            }
            .to_string(),
            "`nope` is not a file or directory"
        );
        assert_eq!(
            WraithError::BadOutput(String::from("dir needed")).to_string(),
            "dir needed"
        );
    }
}
