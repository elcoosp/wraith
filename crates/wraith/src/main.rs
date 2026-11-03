//! Binary entry point. All logic lives in the `wraith` library so it stays
//! unit-testable; this file only wires real process I/O.

use std::io::IsTerminal as _;

fn main() -> std::process::ExitCode {
    let color = std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let argv: Vec<std::ffi::OsString> = std::env::args_os().collect();
    let code = wraith::run(
        &argv,
        &mut std::io::stdin().lock(),
        &mut std::io::stdout().lock(),
        &mut std::io::stderr().lock(),
        color,
    );
    std::process::ExitCode::from(u8::try_from(code).unwrap_or(1))
}
