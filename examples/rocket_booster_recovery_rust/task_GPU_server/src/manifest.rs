use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use syn::{
    ExprUnsafe, ItemExternCrate, ItemForeignMod, ItemMod, ItemUse, Macro, Path as SynPath, UseTree,
    visit::Visit,
};

const MAX_RUST_SOURCE_BYTES: u64 = 2_000_000;
const MAX_RUST_SOURCE_FILES: usize = 64;
const REQUIRED_DIMENSIONS: [&str; 6] = [
    "mechanism_family",
    "intervention_surface",
    "intent",
    "semantic_family",
    "parent_lineage",
    "novelty_axis",
];
const ALLOWED_CHANGED_MODULES: [&str; 10] = [
    "energy_manager",
    "trajectory_guidance",
    "attitude_controller_yz",
    "allocator_yz",
    "fin_effectiveness_model",
    "state_disturbance_estimator",
    "roll_rcs_controller",
    "constraint_governor",
    "terminal_landing_manager",
    "robust_design_validation",
];

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResearchIndependenceAttestation {
    pub prior_run_artifacts_accessed: bool,
    pub external_controller_implementation_accessed: bool,
    pub historical_performance_results_used: bool,
    pub copied_or_translated_prior_solution: bool,
}

impl ResearchIndependenceAttestation {
    pub fn is_clean(&self) -> bool {
        !self.prior_run_artifacts_accessed
            && !self.external_controller_implementation_accessed
            && !self.historical_performance_results_used
            && !self.copied_or_translated_prior_solution
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VariantManifest {
    pub variant_id: String,
    #[serde(default)]
    pub display_name: Option<String>,
    pub method_class: String,
    pub changed_modules: Vec<String>,
    pub design_dimensions: BTreeMap<String, Value>,
    pub research_independence: ResearchIndependenceAttestation,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CandidateAudit {
    pub rust_files: Vec<String>,
    pub rust_source_bytes: u64,
    pub rust_syntax_parsed: bool,
    pub forbidden_runtime_access_detected: bool,
    pub neural_or_rl_dependencies_detected: bool,
    pub research_independence_attested: bool,
    pub controller_sha256: String,
    pub source_tree_sha256: String,
    pub config_sha256: String,
    pub manifest_sha256: String,
}

#[derive(Default)]
struct SyntaxGuard {
    violations: Vec<String>,
}

impl SyntaxGuard {
    fn check_segments(&mut self, segments: &[String]) {
        let forbidden_std = ["fs", "net", "process", "env", "thread", "os"];
        let forbidden_crates = [
            "tch",
            "burn",
            "candle_core",
            "tensorflow",
            "onnxruntime",
            "libc",
        ];
        if segments.first().is_some_and(|root| root == "std")
            && segments
                .get(1)
                .is_some_and(|module| forbidden_std.contains(&module.as_str()))
        {
            self.violations
                .push(format!("forbidden path `{}`", segments.join("::")));
        }
        if segments.first().is_some_and(|root| root == "core")
            && segments.get(1).is_some_and(|module| module == "arch")
        {
            self.violations
                .push(format!("forbidden path `{}`", segments.join("::")));
        }
        if segments
            .first()
            .is_some_and(|root| root == "rocket_booster_recovery_task")
            && !segments
                .get(1)
                .is_some_and(|module| module == "candidate_api")
        {
            self.violations.push(format!(
                "candidate may access only rocket_booster_recovery_task::candidate_api, not `{}`",
                segments.join("::")
            ));
        }
        if segments
            .first()
            .is_some_and(|root| forbidden_crates.contains(&root.as_str()))
        {
            self.violations.push(format!(
                "forbidden dependency path `{}`",
                segments.join("::")
            ));
        }
    }
}

fn collect_use_paths(
    tree: &UseTree,
    prefix: &mut Vec<String>,
    paths: &mut Vec<Vec<String>>,
    renamed: &mut bool,
) {
    match tree {
        UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            collect_use_paths(&path.tree, prefix, paths, renamed);
            prefix.pop();
        }
        UseTree::Name(name) => {
            let mut complete = prefix.clone();
            complete.push(name.ident.to_string());
            paths.push(complete);
        }
        UseTree::Rename(rename) => {
            *renamed = true;
            let mut complete = prefix.clone();
            complete.push(rename.ident.to_string());
            paths.push(complete);
        }
        UseTree::Glob(_) => paths.push(prefix.clone()),
        UseTree::Group(group) => {
            for item in &group.items {
                collect_use_paths(item, prefix, paths, renamed);
            }
        }
    }
}

impl<'ast> Visit<'ast> for SyntaxGuard {
    fn visit_attribute(&mut self, node: &'ast syn::Attribute) {
        let custom_path = node.path().is_ident("path")
            || (node.path().is_ident("cfg_attr")
                && matches!(&node.meta, syn::Meta::List(list) if list.tokens.to_string().contains("path")));
        if custom_path {
            self.violations
                .push("custom module path attribute".to_owned());
        }
        syn::visit::visit_attribute(self, node);
    }

    fn visit_path(&mut self, node: &'ast SynPath) {
        let segments: Vec<String> = node
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect();
        self.check_segments(&segments);
        syn::visit::visit_path(self, node);
    }

    fn visit_item_use(&mut self, node: &'ast ItemUse) {
        let mut paths = Vec::new();
        let mut renamed = false;
        collect_use_paths(&node.tree, &mut Vec::new(), &mut paths, &mut renamed);
        for path in paths {
            self.check_segments(&path);
        }
        syn::visit::visit_item_use(self, node);
    }

    fn visit_expr_unsafe(&mut self, node: &'ast ExprUnsafe) {
        self.violations.push("unsafe block".to_owned());
        syn::visit::visit_expr_unsafe(self, node);
    }

    fn visit_item_foreign_mod(&mut self, node: &'ast ItemForeignMod) {
        self.violations.push("foreign ABI block".to_owned());
        syn::visit::visit_item_foreign_mod(self, node);
    }

    fn visit_item_extern_crate(&mut self, node: &'ast ItemExternCrate) {
        self.violations.push("extern crate declaration".to_owned());
        syn::visit::visit_item_extern_crate(self, node);
    }

    fn visit_item_mod(&mut self, node: &'ast ItemMod) {
        if node
            .attrs
            .iter()
            .any(|attribute| attribute.path().is_ident("path"))
        {
            self.violations.push(format!(
                "custom #[path] module declaration `{}`",
                node.ident
            ));
        }
        syn::visit::visit_item_mod(self, node);
    }

    fn visit_macro(&mut self, node: &'ast Macro) {
        let name = node
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string())
            .unwrap_or_default();
        if matches!(
            name.as_str(),
            "include"
                | "include_bytes"
                | "include_str"
                | "env"
                | "option_env"
                | "asm"
                | "global_asm"
        ) {
            self.violations.push(format!("forbidden macro `{name}!`"));
        }
        syn::visit::visit_macro(self, node);
    }
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

pub fn sha256_named_rust_sources(entries: &[(String, Vec<u8>)]) -> String {
    let mut entries = entries.to_vec();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    let mut hasher = Sha256::new();
    for (relative, bytes) in entries {
        let path_bytes = relative.as_bytes();
        hasher.update((path_bytes.len() as u64).to_le_bytes());
        hasher.update(path_bytes);
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    hex::encode(hasher.finalize())
}

pub fn sha256_rust_source_tree(root: &Path, rust_files: &[String]) -> Result<String> {
    let mut entries = Vec::with_capacity(rust_files.len());
    for relative in rust_files {
        let relative_path = Path::new(&relative);
        ensure!(
            relative_path
                .components()
                .all(|component| matches!(component, std::path::Component::Normal(_))),
            "candidate Rust source path is not a safe relative path: {relative}"
        );
        let bytes = fs::read(root.join(relative_path))
            .with_context(|| format!("read candidate Rust source {relative}"))?;
        entries.push((relative.clone(), bytes));
    }
    Ok(sha256_named_rust_sources(&entries))
}

fn valid_variant_id(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    value.len() <= 128
        && first.is_ascii_alphanumeric()
        && chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '-'))
}

fn collect_files(root: &Path, directory: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(directory)
        .with_context(|| format!("read candidate directory {}", directory.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path)?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "candidate symlinks are forbidden"
        );
        if metadata.is_dir() {
            collect_files(root, &path, out)?;
        } else if metadata.is_file() {
            ensure!(
                path.starts_with(root),
                "candidate file escaped the variant directory"
            );
            out.push(path);
        }
    }
    Ok(())
}

fn scan_rust_source(path: &Path, text: &str) -> Result<()> {
    let parsed = syn::parse_file(text)
        .with_context(|| format!("candidate Rust syntax error in {}", path.display()))?;
    let mut guard = SyntaxGuard::default();
    guard.visit_file(&parsed);

    let forbidden_text = [
        "std::fs",
        "std::net",
        "std::process",
        "std::env",
        "std::thread",
        "std::os",
        "core::arch",
        "Command::",
        "File::open",
        "OpenOptions",
        "TcpStream",
        "UdpSocket",
        "run_candidate_from_env",
        "public_main",
        "rocket_booster_recovery_rust::dataset",
        "rocket_booster_recovery_rust::metrics",
        "rocket_booster_recovery_rust::report",
        "rocket_booster_recovery_rust::rollout",
        "source_banks",
        "evaluation_summary.json",
        "formal_ood_16384",
        "terminal_results.npz",
    ];
    for token in forbidden_text {
        if text.contains(token) {
            guard.violations.push(format!("forbidden token `{token}`"));
        }
    }
    let forbidden_learning = [
        "tch::",
        "burn::",
        "candle_",
        "tensorflow",
        "onnx",
        "reinforcement_learning",
        "policy_network",
    ];
    for token in forbidden_learning {
        if text.to_ascii_lowercase().contains(token) {
            guard
                .violations
                .push(format!("forbidden learning dependency/token `{token}`"));
        }
    }
    if !guard.violations.is_empty() {
        bail!(
            "candidate static contract failed: {}",
            guard.violations.join("; ")
        );
    }
    Ok(())
}

pub fn inspect_variant(variant_dir: &Path) -> Result<(VariantManifest, CandidateAudit)> {
    let variant_dir = variant_dir
        .canonicalize()
        .with_context(|| format!("resolve variant directory {}", variant_dir.display()))?;
    ensure!(variant_dir.is_dir(), "variant path must be a directory");
    let controller = variant_dir.join("controller.rs");
    let config = variant_dir.join("controller_config.json");
    let manifest_path = variant_dir.join("variant.json");
    for required in [&controller, &config, &manifest_path] {
        ensure!(
            required.is_file(),
            "variant is missing {}",
            required.display()
        );
    }

    let mut files = Vec::new();
    collect_files(&variant_dir, &variant_dir, &mut files)?;
    let mut rust_files: Vec<_> = files
        .iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "rs"))
        .collect();
    rust_files.sort();
    ensure!(
        rust_files.contains(&&controller),
        "variant must contain a root Rust source file named controller.rs"
    );
    ensure!(
        rust_files.len() <= MAX_RUST_SOURCE_FILES,
        "variant contains more than {MAX_RUST_SOURCE_FILES} Rust source files"
    );
    for path in &files {
        let relative = path.strip_prefix(&variant_dir)?;
        let required = matches!(
            relative.to_str(),
            Some("controller.rs" | "controller_config.json" | "variant.json")
        );
        ensure!(
            required || path.extension().is_some_and(|ext| ext == "rs"),
            "unsupported candidate file {}; only the three required files and optional .rs modules are allowed",
            relative.display()
        );
    }
    let source_bytes = rust_files
        .iter()
        .try_fold(0_u64, |total, path| -> Result<u64> {
            Ok(total + path.metadata()?.len())
        })?;
    ensure!(
        source_bytes <= MAX_RUST_SOURCE_BYTES,
        "variant Rust source tree exceeds 2 MB"
    );
    for path in &rust_files {
        let source =
            fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        scan_rust_source(path, &source)?;
    }

    let rust_file_names: Vec<String> = rust_files
        .iter()
        .map(|path| {
            path.strip_prefix(&variant_dir)?
                .to_str()
                .map(str::to_owned)
                .context("candidate Rust source path must be UTF-8")
        })
        .collect::<Result<_>>()?;

    let manifest: VariantManifest = serde_json::from_slice(&fs::read(&manifest_path)?)
        .with_context(|| format!("parse {}", manifest_path.display()))?;
    ensure!(
        valid_variant_id(&manifest.variant_id),
        "variant_id is missing or unsafe"
    );
    ensure!(
        manifest.method_class == "deterministic_classical_control",
        "method_class must be deterministic_classical_control"
    );
    ensure!(
        manifest.research_independence.is_clean(),
        "research independence attestation is not clean; candidate is ineligible"
    );
    let allowed: BTreeSet<_> = ALLOWED_CHANGED_MODULES.into_iter().collect();
    let unknown: Vec<_> = manifest
        .changed_modules
        .iter()
        .filter(|name| !allowed.contains(name.as_str()))
        .cloned()
        .collect();
    ensure!(
        unknown.is_empty(),
        "changed_modules outside allowed surface: {}",
        unknown.join(", ")
    );
    for dimension in REQUIRED_DIMENSIONS {
        let value = manifest
            .design_dimensions
            .get(dimension)
            .with_context(|| format!("variant design_dimensions missing: {dimension}"))?;
        ensure!(
            value.as_str().is_some_and(|text| !text.trim().is_empty()),
            "variant design dimension {dimension} must be a non-empty string"
        );
    }

    let audit = CandidateAudit {
        rust_files: rust_file_names.clone(),
        rust_source_bytes: source_bytes,
        rust_syntax_parsed: true,
        forbidden_runtime_access_detected: false,
        neural_or_rl_dependencies_detected: false,
        research_independence_attested: true,
        controller_sha256: sha256_file(&controller)?,
        source_tree_sha256: sha256_rust_source_tree(&variant_dir, &rust_file_names)?,
        config_sha256: sha256_file(&config)?,
        manifest_sha256: sha256_file(&manifest_path)?,
    };
    Ok((manifest, audit))
}

pub fn variant_id_hint(variant_dir: &Path) -> String {
    let fallback = variant_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown_variant")
        .to_owned();
    let path = variant_dir.join("variant.json");
    let Ok(bytes) = fs::read(path) else {
        return fallback;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return fallback;
    };
    value["variant_id"]
        .as_str()
        .filter(|id| valid_variant_id(id))
        .unwrap_or(&fallback)
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn baseline_variant_is_contract_clean_rust() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/baseline");
        let (manifest, audit) = inspect_variant(&root).unwrap();
        assert_eq!(
            manifest.variant_id,
            "rocket_booster_recovery_rust_v2_first_contact_7000kg_baseline"
        );
        assert_eq!(audit.rust_files, ["controller.rs"]);
        assert!(!audit.forbidden_runtime_access_detected);
        assert!(audit.research_independence_attested);
    }

    #[test]
    fn static_scan_rejects_filesystem_and_unsafe_access() {
        let error = scan_rust_source(
            Path::new("controller.rs"),
            "use std::{fs}; fn bad() { let _ = fs::read(\"secret\"); unsafe {} }",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("unsafe block"));
        assert!(error.contains("std::fs"));
    }

    #[test]
    fn multi_file_candidate_tree_is_audited_and_path_escape_is_rejected() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "rocket-booster-recovery-rust-manifest-test-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("modules")).unwrap();
        fs::write(
            root.join("controller.rs"),
            "mod modules; pub fn marker() {}\n",
        )
        .unwrap();
        fs::write(
            root.join("modules/mod.rs"),
            "pub fn helper() -> f32 { 1.0 }\n",
        )
        .unwrap();
        fs::write(root.join("controller_config.json"), "{}\n").unwrap();
        fs::write(
            root.join("variant.json"),
            r#"{
              "variant_id":"multi_file_test",
              "method_class":"deterministic_classical_control",
              "changed_modules":["energy_manager"],
              "design_dimensions":{
                "mechanism_family":"analytic_energy",
                "intervention_surface":"energy_manager",
                "intent":"repair",
                "semantic_family":"multi_file_fixture",
                "parent_lineage":"baseline",
                "novelty_axis":"module_staging"
              },
              "research_independence":{
                "prior_run_artifacts_accessed":false,
                "external_controller_implementation_accessed":false,
                "historical_performance_results_used":false,
                "copied_or_translated_prior_solution":false
              }
            }"#,
        )
        .unwrap();
        let (_, audit) = inspect_variant(&root).unwrap();
        assert_eq!(audit.rust_files, ["controller.rs", "modules/mod.rs"]);
        assert_eq!(
            audit.source_tree_sha256,
            sha256_rust_source_tree(&root, &audit.rust_files).unwrap()
        );

        let manifest_path = root.join("variant.json");
        let mut contaminated: Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        contaminated["research_independence"]["copied_or_translated_prior_solution"] =
            Value::Bool(true);
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&contaminated).unwrap(),
        )
        .unwrap();
        let independence_error = inspect_variant(&root).unwrap_err().to_string();
        assert!(independence_error.contains("research independence attestation"));

        let error = scan_rust_source(
            Path::new("controller.rs"),
            "#[path = \"../secret.rs\"] mod secret;",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("custom #[path]"));

        let cfg_attr_error = scan_rust_source(
            Path::new("controller.rs"),
            "#[cfg_attr(all(), path = \"../secret.rs\")] mod secret;",
        )
        .unwrap_err()
        .to_string();
        assert!(cfg_attr_error.contains("custom module path attribute"));
        fs::remove_dir_all(root).unwrap();
    }
}
