import java.util.Random;

public class Matrix {
	static int n = 1024;
	static double[][] a = new double[n][n];
	static double[][] b = new double[n][n];
	static double[][] c = new double[n][n];

	public static void main(String[] args) {
		Random random = new Random();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				a[i][j] = random.nextDouble();
				b[i][j] = random.nextDouble();
				c[i][j] = 0;
			}
		}

		long start = System.currentTimeMillis();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				for (int k = 0; k < n; k++) {
					c[i][j] += a[i][k] * b[k][j];
				}
			}
		}
		long stop = System.currentTimeMillis();
		System.out.println((stop-start) * 1e-3);
	}
}
