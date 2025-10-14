use std::time::SystemTime;
use rand::{Rng, thread_rng};

fn main() {
    let n = 1024;
    let mut a = vec![vec![0.0_f64; n]; n];
    let mut b = vec![vec![0.0_f64; n]; n];
    let mut c = vec![vec![0.0_f64; n]; n];

    let mut rng = thread_rng();

    // Inicializar las matrices con números aleatorios
    for i in 0..n {
        for j in 0..n {
            a[i][j] = rng.gen::<f64>();
            b[i][j] = rng.gen::<f64>();
        }
    }

    // Medir el tiempo de multiplicación
    let start = SystemTime::now();

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    let elapsed = start.elapsed().expect("Error al medir el tiempo");

    println!("Tiempo transcurrido: {:.2?}", elapsed);
}
