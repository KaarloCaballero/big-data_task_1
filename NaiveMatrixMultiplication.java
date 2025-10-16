import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.lang.management.ManagementFactory;
import com.sun.management.OperatingSystemMXBean;

public class NaiveMatrixMultiplication {

    // --- Parameters ---
    static int[] matrixSizes = {10, 100}; // Adjust sizes as needed
    static int iterations = 10;           // Adjust for testing
    static int pauseEvery = 20;           // Pause every 20 iterations
    static int pauseDuration = 10;        // Pause duration in seconds
    static String language = "Java";
    static String matrixDir = "matrices";
    static String csvFile = "results/java_results.csv";

    // --- System monitoring ---
    static OperatingSystemMXBean osBean = 
        (OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
    static Runtime runtime = Runtime.getRuntime();

    // --- Read binary matrix ---
    public static int[][] readMatrixFromBinary(String filename, int size) throws IOException {
        byte[] bytes = Files.readAllBytes(Paths.get(filename));
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int[][] matrix = new int[size][size];
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                matrix[i][j] = buffer.getInt();
        System.out.println("[OK] Loaded matrix from '" + filename + "' (" + size + "x" + size + ")");
        return matrix;
    }

    // --- Naive matrix multiplication ---
    public static double naiveMatrixMultiplication(int[][] A, int[][] B, int n) {
        int[][] C = new int[n][n];
        long start = System.nanoTime();
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < n; k++)
                    C[i][j] += A[i][k] * B[k][j];
        long end = System.nanoTime();
        return (end - start) / 1e9; // Convert to seconds
    }

    // --- Warm-up ---
    public static void warmUp(int[][] A, int[][] B, int size, int iterations, int pauseSeconds) throws InterruptedException {
        System.out.println("\n=== Warm-up: " + iterations + " iterations for size " + size + "x" + size + " ===");
        for (int i = 1; i <= iterations; i++) {
            naiveMatrixMultiplication(A, B, size);
            System.out.println("[OK] Warm-up iteration " + i + " completed");
            TimeUnit.SECONDS.sleep(pauseSeconds);
        }
    }

    // --- Save results to CSV ---
    public static void saveResultsToCSV(List<String[]> results) throws IOException {
        File file = new File(csvFile);
        file.getParentFile().mkdirs();
        try (PrintWriter writer = new PrintWriter(file)) {
            writer.println("Size,Matrix A File,Matrix B File,"
                + "Mean Time (s),Median Time (s),Std Dev (s),"
                + "Mean CPU (%),Median CPU (%),Std Dev CPU (%),"
                + "Mean Memory (MB),Median Memory (MB),Std Dev Memory (MB),Language");
            for (String[] row : results)
                writer.println(String.join(",", row));
        }
        System.out.println("\n[OK] All results saved to " + csvFile);
    }

    // --- Statistics ---
    public static double mean(List<Double> data) {
        return data.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }

    public static double median(List<Double> data) {
        List<Double> sorted = data.stream().sorted().collect(Collectors.toList());
        int n = sorted.size();
        if (n % 2 == 0)
            return (sorted.get(n/2 - 1) + sorted.get(n/2)) / 2.0;
        else
            return sorted.get(n/2);
    }

    public static double std(List<Double> data, double mean) {
        double sum = 0;
        for (double x : data) sum += Math.pow(x - mean, 2);
        return Math.sqrt(sum / data.size());
    }

    // --- Get CPU and memory usage ---
    public static double getProcessCpuLoad() {
        return osBean.getProcessCpuLoad() * 100; // percent
    }

    public static double getMemoryUsageMB() {
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024.0 * 1024.0);
    }

    // --- Main ---
    public static void main(String[] args) throws Exception {
        List<String[]> results = new ArrayList<>();

        int maxSize = Arrays.stream(matrixSizes).max().getAsInt();
        int[][] matrixAWarmUp = readMatrixFromBinary(matrixDir + "/A_" + maxSize + ".bin", maxSize);
        int[][] matrixBWarmUp = readMatrixFromBinary(matrixDir + "/B_" + maxSize + ".bin", maxSize);
        warmUp(matrixAWarmUp, matrixBWarmUp, maxSize, 5, 2);

        for (int size : matrixSizes) {
            String fileA = matrixDir + "/A_" + size + ".bin";
            String fileB = matrixDir + "/B_" + size + ".bin";

            System.out.println("\n=== Processing matrices of size " + size + "x" + size + " ===");
            int[][] matrixA = readMatrixFromBinary(fileA, size);
            int[][] matrixB = readMatrixFromBinary(fileB, size);

            List<Double> times = new ArrayList<>();
            List<Double> cpuUsages = new ArrayList<>();
            List<Double> memoryUsages = new ArrayList<>();

            System.out.println("\n=== Multiplying matrices of size " + size + "x" + size + " ===");
            System.out.println("Running " + iterations + " iterations with a " + pauseDuration + "s pause every " + pauseEvery + " iterations...");

            for (int i = 1; i <= iterations; i++) {
                double elapsed = naiveMatrixMultiplication(matrixA, matrixB, size);
                double cpu = getProcessCpuLoad();
                double mem = getMemoryUsageMB();

                times.add(elapsed);
                cpuUsages.add(cpu);
                memoryUsages.add(mem);

                if (i % pauseEvery == 0 && i != iterations) {
                    System.out.println("[PAUSE] Pausing for " + pauseDuration + " seconds to cool off the CPU...");
                    TimeUnit.SECONDS.sleep(pauseDuration);
                }
            }

            // Compute statistics
            double meanTime = mean(times), medianTime = median(times), stdTime = std(times, meanTime);
            double meanCPU = mean(cpuUsages), medianCPU = median(cpuUsages), stdCPU = std(cpuUsages, meanCPU);
            double meanMem = mean(memoryUsages), medianMem = median(memoryUsages), stdMem = std(memoryUsages, meanMem);

            System.out.printf("[OK] Stats for size %d: mean_time=%.6f, mean_cpu=%.2f%%, mean_mem=%.2fMB%n", size, meanTime, meanCPU, meanMem);

            results.add(new String[]{
                String.valueOf(size), fileA, fileB,
                String.valueOf(meanTime), String.valueOf(medianTime), String.valueOf(stdTime),
                String.valueOf(meanCPU), String.valueOf(medianCPU), String.valueOf(stdCPU),
                String.valueOf(meanMem), String.valueOf(medianMem), String.valueOf(stdMem),
                language
            });
        }

        saveResultsToCSV(results);
        System.out.println("\n[OK] Process completed successfully");
    }
}
