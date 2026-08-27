fn main() {
    if let Err(error) = rocket_booster_recovery_task::public_main() {
        eprintln!("rocket-booster-recovery-rust task evaluator failed: {error:#}");
        std::process::exit(2);
    }
}
