fn main() {
    if let Err(error) = imat::run() {
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}
