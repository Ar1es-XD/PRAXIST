use std::{
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::plant::State;

pub const COMPLETE_ROWS: usize = 12_288;
pub const ROWS_PER_COMPLETE_SOURCE: usize = 4_096;
pub const INITIAL_MASS_KG: f32 = 29_200.0;
pub const INITIAL_FUEL_KG: f32 = 7_000.0;
pub const MASS_EMPTY_KG: f32 = 22_200.0;
pub const SOURCE_ROWS_DIGEST: &str =
    "2694073419a116a12c611f8e952cf399b0a2172ea0c93aa8c6bac36ceee9b78c";

pub const SOURCE_BANKS: [(&str, &str, &str); 3] = [
    (
        "nominal_unseen",
        "data/source_banks/nominal_unseen_40960.npz",
        "674d119f8f0cd36c2553f0cc9134ec23886fd00d1a34396d515d957132311a3d",
    ),
    (
        "near_ood",
        "data/source_banks/near_ood_easy_velocity_40960.npz",
        "d35803608dd3dedbbd4db76d9b39aebe37c2eedbdba4ad9ec0dcd94a690d7730",
    ),
    (
        "hard_ood",
        "data/source_banks/hard_ood_fast_outer_annulus_40960.npz",
        "55d3e0054ef5d225b78f6558a4e15a3a908bedb84daabb7cc66b8e6579fd297c",
    ),
];

pub const DEVELOPMENT_PATH: &str = "data/development_ood_2048.npz";
pub const DEVELOPMENT_HASH: &str =
    "00539ee4e538fab8a65e82ae737289e5b42aa9dfae70da9da1cec5e7c8871f94";
pub const SELECTION_PATH: &str = "data/complete_source_rows_le_i32.bin";
pub const SELECTION_HASH: &str = "4f7ab2fcdefdf7adfcb6f9a4528e9eb53019a4323e0a606f68e5ba98505b80bc";

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Canary,
    Development,
    Complete,
}

impl Mode {
    pub fn expected_rows(self) -> usize {
        match self {
            Self::Canary => 1,
            Self::Development => 2_048,
            Self::Complete => COMPLETE_ROWS,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Canary => "canary",
            Self::Development => "development",
            Self::Complete => "complete",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub states: Vec<State>,
    pub source_ids: Vec<u8>,
    pub source_rows: Vec<i32>,
    pub source_names: Vec<String>,
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut digest = Sha256::new();
    std::io::copy(&mut file, &mut digest).with_context(|| format!("hash {}", path.display()))?;
    Ok(hex::encode(digest.finalize()))
}

pub fn attest_assets(root: &Path) -> Result<Vec<(String, String)>> {
    let mut assets = Vec::new();
    for (_, relative, expected) in SOURCE_BANKS {
        attest_one(root, relative, expected, &mut assets)?;
    }
    attest_one(root, DEVELOPMENT_PATH, DEVELOPMENT_HASH, &mut assets)?;
    attest_one(root, SELECTION_PATH, SELECTION_HASH, &mut assets)?;
    Ok(assets)
}

fn attest_one(
    root: &Path,
    relative: &str,
    expected: &str,
    observed: &mut Vec<(String, String)>,
) -> Result<()> {
    let path = root.join(relative);
    ensure!(
        path.is_file(),
        "required asset is missing: {}",
        path.display()
    );
    let actual = sha256_file(&path)?;
    ensure!(
        actual == expected,
        "asset hash mismatch for {relative}: expected {expected}, got {actual}"
    );
    observed.push((relative.to_owned(), actual));
    Ok(())
}

fn read_state_array(path: &Path) -> Result<Array2<f32>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut npz = NpzReader::new(file).with_context(|| format!("parse {}", path.display()))?;
    npz.by_name("state")
        .or_else(|_| npz.by_name("state.npy"))
        .with_context(|| format!("read state array from {}", path.display()))
}

fn row_to_state(array: &Array2<f32>, row: usize) -> Result<State> {
    ensure!(
        array.ncols() == 16,
        "state array has {} columns, expected 16",
        array.ncols()
    );
    ensure!(
        row < array.nrows(),
        "source row {row} is outside {} rows",
        array.nrows()
    );
    let slice = array.row(row);
    let mut state = [0.0; 16];
    for (target, value) in state.iter_mut().zip(slice.iter()) {
        *target = *value;
    }
    state[13] = INITIAL_MASS_KG;
    Ok(state)
}

fn complete_selection(root: &Path) -> Result<Vec<i32>> {
    let bytes = std::fs::read(root.join(SELECTION_PATH))?;
    ensure!(
        bytes.len() == COMPLETE_ROWS * 4,
        "selection asset has invalid length"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("four byte chunk")))
        .collect())
}

fn load_complete(root: &Path) -> Result<Dataset> {
    let selection = complete_selection(root)?;
    let mut states = Vec::with_capacity(COMPLETE_ROWS);
    let mut source_ids = Vec::with_capacity(COMPLETE_ROWS);
    for (source_id, (_, relative, _)) in SOURCE_BANKS.iter().enumerate() {
        let array = read_state_array(&root.join(relative))?;
        ensure!(
            array.nrows() == 40_960,
            "{relative} has {} rows",
            array.nrows()
        );
        let range =
            source_id * ROWS_PER_COMPLETE_SOURCE..(source_id + 1) * ROWS_PER_COMPLETE_SOURCE;
        for &source_row in &selection[range] {
            states.push(row_to_state(&array, source_row as usize)?);
            source_ids.push(source_id as u8);
        }
    }
    Ok(Dataset {
        states,
        source_ids,
        source_rows: selection,
        source_names: SOURCE_BANKS
            .iter()
            .map(|(name, _, _)| (*name).to_owned())
            .collect(),
    })
}

fn npz_by_name<T, D>(npz: &mut NpzReader<File>, name: &str) -> Result<ndarray::Array<T, D>>
where
    T: ndarray_npy::ReadableElement,
    D: ndarray::Dimension,
{
    npz.by_name(name)
        .or_else(|_| npz.by_name(&format!("{name}.npy")))
        .with_context(|| format!("read {name} from development bank"))
}

fn load_development(root: &Path, limit: usize) -> Result<Dataset> {
    let path = root.join(DEVELOPMENT_PATH);
    let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
    let mut npz = NpzReader::new(file).with_context(|| format!("parse {}", path.display()))?;
    let state: Array2<f32> = npz_by_name(&mut npz, "state")?;
    let source: Array1<i8> = npz_by_name(&mut npz, "source_id")?;
    let rows: Array1<i32> = npz_by_name(&mut npz, "source_row")?;
    ensure!(
        state.nrows() == source.len() && state.nrows() == rows.len(),
        "development arrays disagree in length"
    );
    ensure!(
        limit <= state.nrows(),
        "requested {limit} development rows, only {} exist",
        state.nrows()
    );
    let mut states = Vec::with_capacity(limit);
    for index in 0..limit {
        states.push(row_to_state(&state, index)?);
    }
    Ok(Dataset {
        states,
        source_ids: source
            .iter()
            .take(limit)
            .map(|&value| value as u8)
            .collect(),
        source_rows: rows.iter().take(limit).copied().collect(),
        source_names: vec!["near_ood".to_owned(), "hard_ood".to_owned()],
    })
}

pub fn load(root: &Path, mode: Mode) -> Result<Dataset> {
    let dataset = match mode {
        Mode::Complete => load_complete(root)?,
        Mode::Development => load_development(root, 2_048)?,
        Mode::Canary => load_development(root, 1)?,
    };
    if dataset.states.len() != mode.expected_rows() {
        bail!(
            "{} mode loaded {} rows, expected {}",
            mode.as_str(),
            dataset.states.len(),
            mode.expected_rows()
        );
    }
    Ok(dataset)
}

pub fn asset_path(root: &Path, relative: &str) -> PathBuf {
    root.join(relative)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    #[test]
    fn complete_selection_has_reference_digest() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let rows = complete_selection(root).unwrap();
        assert_eq!(rows.len(), COMPLETE_ROWS);
        let json = serde_json::to_vec(&rows).unwrap();
        assert_eq!(hex::encode(Sha256::digest(json)), SOURCE_ROWS_DIGEST);
    }

    #[test]
    fn canary_loads_with_mass_override() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let data = load(root, Mode::Canary).unwrap();
        assert_eq!(data.states.len(), 1);
        assert_eq!(data.states[0][13], INITIAL_MASS_KG);
    }
}
