use std::fs::{self, File};
use std::convert::TryInto;
use std::io::{BufWriter, Write, Read};
use std::path::Path;
use std::thread::sleep;
use std::time::{Duration, Instant};

/// Matrix Multiplication Benchmark Script (Rust version)
///
/// Performs naive matrix multiplication on pre-generated binary matrices.
/// For each matrix size, it executes the multiplication 100 times,
/// pausing for 10 seconds every 20 iterations to cool off the CPU.
/// Measures mean, median, and std deviation and saves results to a CSV file.

// === Parameters ===
const MATRIX_SIZES: [usize; 4] = [10, 100, 1000, 10000];
const ITERATIONS: usize = 100;
const PAUSE_EVERY: usize = 20;
const PAUSE_DURATION: u64 = 10;
const CSV_FILE: &str = "results/rust_results.csv";
const LANGUAGE: &str = "Rust";
const MATRIX_DIR: &str = "matrices";

// --- Read matrix from binary ---
fn read_matrix_from_binary(filename: &str, size: usize) -> Vec<Vec<i32>> {
    let path = Path::new(filename);
    let mut file = File::open(path)
        .unwrap_or_else(|_| panic!("❌ Couldn't open file '{}'", filename));

    let num_elements = size * size;
    let mut buffer = vec![0i32; num_elements];

    let bytes_to_read = num_elements * std::mem::size_of::<i32>();
    let mut raw_bytes = vec![0u8; bytes_to_read];

    file.read_exact(&mut raw_bytes)
        .unwrap_or_else(|_| panic!("❌ Could not read matrix from '{}'", filename));

    // Convert bytes to i32
    for i in 0..num_elements {
        buffer[i] = i32::from_le_bytes(raw_bytes[i * 4..i * 4 + 4].try_into().unwrap());
    }

    // Convert flat buffer to 2D Vec<Vec<i32>>
    let matrix: Vec<Vec<i32>> = buffer
        .chunks(size)
        .map(|chunk| chunk.to_vec())
        .collect();

    println!("✅ Loaded matrix from '{}' ({}x{})", filename, size, size);
    matrix
}

// --- Naive matrix multiplication ---
fn naive_matrix_multiplication(a: &Vec<Vec<i32>>, b: &Vec<Vec<i32>>, n: usize) -> f64 {
    let mut c = vec![vec![0i32; n]; n];

    let start = Instant::now();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    let duration = start.elapsed();
    duration.as_secs_f64()
}

// --- Warm-up ---
fn warm_up(a: &Vec<Vec<i32>>, b: &Vec<Vec<i32>>, size: usize, iterations: usize, pause: u64) {
    println!("\n=== Warm-up: {} iterations for size {}x{} ===", iterations, size, size);
    for i in 1..=iterations {
        naive_matrix_multiplication(a, b, size);
        println!("✅ Warm-up iteration {} completed", i);
        sleep(Duration::from_secs(pause));
    }
}

// --- Compute statistics ---
fn mean(data: &Vec<f64>) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn median(data: &mut Vec<f64>) -> f64 {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = data.len();
    if n % 2 == 0 {
        (data[n / 2 - 1] + data[n / 2]) / 2.0
    } else {
        data[n / 2]
    }
}

fn std_dev(data: &Vec<f64>, mean_val: f64) -> f64 {
    let variance = data.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

// --- Save results to CSV ---
fn save_results_to_csv(results: &Vec<(usize, String, String, f64, f64, f64, String)>) {
    let path = Path::new(CSV_FILE);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("❌ Couldn't create results directory");
    }

    println!("\nSaving all results to {}...", CSV_FILE);
    let file = File::create(path).expect("❌ Couldn't open CSV file");
    let mut writer = BufWriter::new(file);

    writeln!(
        writer,
        "Size,Matrix A File,Matrix B File,Mean Time (s),Median Time (s),Std Dev (s),Language"
    )
    .unwrap();

    for (size, file_a, file_b, mean, median, std, lang) in results {
        writeln!(
            writer,
            "{},{},{},{},{},{},{}",
            size, file_a, file_b, mean, median, std, lang
        )
        .unwrap();
    }

    println!("✅ All results saved successfully");
}

// --- Main ---
fn main() {

    let mut results: Vec<(usize, String, String, f64, f64, f64, String)> = Vec::new();

    // Warm-up with largest matrix
    let max_size = *MATRIX_SIZES.iter().max().unwrap();
    let file_a_warm = format!("{}/A_{}.bin", MATRIX_DIR, max_size);
    let file_b_warm = format!("{}/B_{}.bin", MATRIX_DIR, max_size);
    let matrix_a_warm = read_matrix_from_binary(&file_a_warm, max_size);
    let matrix_b_warm = read_matrix_from_binary(&file_b_warm, max_size);
    warm_up(&matrix_a_warm, &matrix_b_warm, max_size, 5, 2);

    // Benchmark all sizes
    for &size in MATRIX_SIZES.iter() {
        let file_a = format!("{}/A_{}.bin", MATRIX_DIR, size);
        let file_b = format!("{}/B_{}.bin", MATRIX_DIR, size);

        println!("\n=== Processing matrices of size {}x{} ===", size, size);
        let matrix_a = read_matrix_from_binary(&file_a, size);
        let matrix_b = read_matrix_from_binary(&file_b, size);

        let mut times: Vec<f64> = Vec::with_capacity(ITERATIONS);
        println!(
            "\n=== Multiplicating matrices of size {}x{} ===",
            size, size
        );
        println!(
            "Running {} iterations with a {}s pause every {} iterations...",
            ITERATIONS, PAUSE_DURATION, PAUSE_EVERY
        );

        for i in 1..=ITERATIONS {
            let elapsed = naive_matrix_multiplication(&matrix_a, &matrix_b, size);
            times.push(elapsed);

            if i % PAUSE_EVERY == 0 && i != ITERATIONS {
                println!("💤 Pausing for {} seconds to cool off the CPU...", PAUSE_DURATION);
                sleep(Duration::from_secs(PAUSE_DURATION));
            }
        }

        let mean_time = mean(&times);
        let mut sorted_times = times.clone();
        let median_time = median(&mut sorted_times);
        let std_time = std_dev(&times, mean_time);

        println!(
            "✅ Stats for size {}: mean={:.6}, median={:.6}, std={:.6}",
            size, mean_time, median_time, std_time
        );

        results.push((
            size,
            file_a.clone(),
            file_b.clone(),
            mean_time,
            median_time,
            std_time,
            LANGUAGE.to_string(),
        ));
    }

    save_results_to_csv(&results);
    println!("\n✅ Process completed successfully");
}
