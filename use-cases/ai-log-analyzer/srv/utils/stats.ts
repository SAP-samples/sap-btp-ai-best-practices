export function calculateStats(data: { count: number }[]) {
    const counts = data.map(d => d.count).sort((a, b) => a - b);
    const n = counts.length;
  
    if (n === 0) {
      throw new Error("No data to calculate stats");
    }
  
    const mean = counts.reduce((sum, c) => sum + c, 0) / n;
  
    const median =
      n % 2 === 0
        ? (counts[n / 2 - 1] + counts[n / 2]) / 2
        : counts[Math.floor(n / 2)];
  
    const stddev = Math.sqrt(
      counts.reduce((sum, c) => sum + Math.pow(c - mean, 2), 0) / n
    );
  
    const min = counts[0];
    const max = counts[n - 1];
  
    const percentile = (p: number) => {
      const idx = p * (n - 1);
      const lower = Math.floor(idx);
      const upper = Math.ceil(idx);
      const weight = idx - lower;
      return counts[lower] * (1 - weight) + counts[upper] * weight;
    };
  
    return {
      total: n,
      mean,
      median,
      stddev,
      min,
      max,
      p95: percentile(0.95),
      p99: percentile(0.99)
    };
  }