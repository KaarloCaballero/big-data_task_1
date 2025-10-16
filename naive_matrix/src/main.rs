use std::fs::{self, File};
use std::convert::TryInto;
use std::io::{BufWriter, Write, Read};
use std::path::Path;
use std::thread::sleep;
use std::time::{Duration, Instant};

use sysinfo::{System, SystemExt, ProcessExt};

// --- Parameters ---
const MATRIX_SIZES: [usize; 2] = [10, 100]; // adjust as needed
const ITERATIONS: usize = 10;
const PAUSE_EVERY: usize = 20;
const PAUSE_DURATION: u64 = 10;
const CSV_FILE: &str = "results/rust_results.csv";
const LANGUAGE: &str = "Rust";
const MATRIX_DIR: &str = "matrices";

// --- Read matrix from binary ---
fn read_matrix_from_binary(filename: &str, size: usize) -> Vec<Vec<i32>> {
    let mut file = File::open(filename)
        .unwrap_or_else(|_| panic!("❌ Couldn't open file '{}'", filename));

    let num_elements = size * size;
    let mut raw_bytes = vec![0u8; num_elements * 4];
    file.read_exact(&mut raw_bytes)
        .unwrap_or_else(|_| panic!("❌ Could not read matrix from '{}'", filename));

    let buffer: Vec<i32> = raw_bytes
        .chunks(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect();

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
    start.elapsed().as_secs_f64()
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

// --- Statistics ---
fn mean(data: &Vec<f64>) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn median(data: &mut Vec<f64>) -> f64 {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = data.len();
    if n % 2 == 0 { (data[n/2 -1] + data[n/2]) / 2.0 } else { data[n/2] }
}

fn std_dev(data: &Vec<f64>, mean_val: f64) -> f64 {
    let variance = data.iter().map(|x| (x - mean_val).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

// --- Save results to CSV ---
fn save_results_to_csv(results: &Vec<(usize,String,String,f64,f64,f64,f64,f64,f64,f64,f64,f64,String)>) {
    let path = Path::new(CSV_FILE);
    if let Some(parent) = path.parent() { fs::create_dir_all(parent).unwrap(); }

    println!("\nSaving all results to {}...", CSV_FILE);
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);

    writeln!(writer,
        "Size,Matrix A File,Matrix B File,Mean Time (s),Median Time (s),Std Dev (s),Mean CPU (%),Median CPU (%),Std CPU (%),Mean Memory (MB),Median Memory (MB),Std Memory (MB),Language"
    ).unwrap();

    for r in results {
        writeln!(writer,
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}",
            r.0,r.1,r.2,r.3,r.4,r.5,r.6,r.7,r.8,r.9,r.10,r.11,r.12
        ).unwrap();
    }
    println!("✅ All results saved successfully");
}

// --- Main ---
fn main() {
    let mut sys = System::new_all();
    let pid = sysinfo::get_current_pid().unwrap();
    let mut results: Vec<(usize,String,String,f64,f64,f64,f64,f64,f64,f64,f64,f64,String)> = Vec::new();

    // Warm-up with largest matrix
    let max_size = *MATRIX_SIZES.iter().max().unwrap();
    let file_a_warm = format!("{}/A_{}.bin", MATRIX_DIR, max_size);
    let file_b_warm = format!("{}/B_{}.bin", MATRIX_DIR, max_size);
    let matrix_a_warm = read_matrix_from_binary(&file_a_warm, max_size);
    let matrix_b_warm = read_matrix_from_binary(&file_b_warm, max_size);
    warm_up(&matrix_a_warm, &matrix_b_warm, max_size, 5, 2);

    for &size in MATRIX_SIZES.iter() {
        let file_a = format!("{}/A_{}.bin", MATRIX_DIR, size);
        let file_b = format!("{}/B_{}.bin", MATRIX_DIR, size);
        println!("\n=== Processing matrices of size {}x{} ===", size, size);

        let matrix_a = read_matrix_from_binary(&file_a, size);
        let matrix_b = read_matrix_from_binary(&file_b, size);

        let mut times: Vec<f64> = Vec::with_capacity(ITERATIONS);
        let mut cpu_usage: Vec<f64> = Vec::with_capacity(ITERATIONS);
        let mut mem_usage: Vec<f64> = Vec::with_capacity(ITERATIONS);

        println!("Running {} iterations with a {}s pause every {} iterations...", ITERATIONS, PAUSE_DURATION, PAUSE_EVERY);

        for i in 0..ITERATIONS {
            sys.refresh_process(pid);
            let proc = sys.process(pid).unwrap();

            let start_mem = proc.memory(); // KB
            let start = Instant::now();
            naive_matrix_multiplication(&matrix_a, &matrix_b, size);
            let elapsed = start.elapsed().as_secs_f64();
            sys.refresh_process(pid);
            let proc2 = sys.process(pid).unwrap();
            let end_mem = proc2.memory(); // KB
            let cpu = proc2.cpu_usage(); // %

            times.push(elapsed);
            cpu_usage.push(cpu as f64);
            mem_usage.push((end_mem - start_mem) as f64 / 1024.0); // MB

            if (i+1) % PAUSE_EVERY == 0 && (i+1)!=ITERATIONS {
                println!("💤 Pausing for {} seconds...", PAUSE_DURATION);
                sleep(Duration::from_secs(PAUSE_DURATION));
            }
        }

        let mean_time = mean(&times);
        let median_time = median(&mut times.clone());
        let std_time = std_dev(&times, mean_time);

        let mean_cpu = mean(&cpu_usage);
        let median_cpu = median(&mut cpu_usage.clone());
        let std_cpu = std_dev(&cpu_usage, mean_cpu);

        let mean_mem = mean(&mem_usage);
        let median_mem = median(&mut mem_usage.clone());
        let std_mem = std_dev(&mem_usage, mean_mem);

        println!("✅ Stats for size {}: mean_time={:.6}, mean_cpu={:.2}, mean_mem={:.2}", size, mean_time, mean_cpu, mean_mem);

        results.push((
            size, file_a.clone(), file_b.clone(),
            mean_time, median_time, std_time,
            mean_cpu, median_cpu, std_cpu,
            mean_mem, median_mem, std_mem,
            LANGUAGE.to_string()
        ));
    }

    save_results_to_csv(&results);
    println!("\n✅ Process completed successfully");
}
