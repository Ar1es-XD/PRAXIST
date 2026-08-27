#![recursion_limit = "256"]

//! Frozen, CPU-only Praxist task harness for Rocket Booster Recovery controller research.

pub mod candidate_api;

mod evaluator;
mod manifest;
mod rollout;

pub use evaluator::{public_main, run_candidate_from_env};
