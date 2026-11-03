//! # wraith-core
//!
//! The conversion pipeline behind [`wraith`](https://github.com/wraith-rs/wraith):
//! PNG bytes in, optimised SVG out, stage-by-stage timings along the way.
//!
//! ```no_run
//! use wraith_core::{convert, Settings};
//!
//! let png: Vec<u8> = std::fs::read("input.png").expect("read png");
//! let out = convert(&png, &Settings::new()).expect("convert");
//! std::fs::write("output.svg", out.svg).expect("write svg");
//! println!("saved {:.1}%", out.stats.reduction() * 100.0);
//! ```
//!
//! Pipeline stages, in order:
//! 1. [`preprocess`] — decode PNG, optional luminance threshold,
//! 2. [`trace`] — cluster tracing (vtracer),
//! 3. [`svgout`] — canonical serialization with fill hoisting,
//! 4. [`minify`] — OXVG job set.
#![allow(clippy::module_name_repetitions)]

pub mod error;
pub mod minify;
pub mod pipeline;
pub mod preprocess;
pub mod settings;
pub mod stats;
pub mod svgout;
pub mod trace;

pub use error::{Result, WraithError};
pub use pipeline::{Output, convert};
pub use settings::{Minify, Mode, Quality, Settings};
pub use stats::{ConvertStats, Stage, Stopwatch};
