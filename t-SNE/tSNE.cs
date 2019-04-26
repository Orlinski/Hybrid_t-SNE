using System;
using System.Collections.Generic;
using System.Threading.Tasks;


namespace Hybrid_tSNE
{
    public enum RepulsionMethods
    {
        auto = 0,
        barnes_hut,
        fft
    }

    public class tSNE
    {
        private float[][] data;
        private readonly int N;
        private readonly int D;
        private int Dims;
        public bool verbose = false;

        private readonly Gradient gradient = new Gradient();

        /// <summary>Result initialization configuration.</summary>
        public InitializationConfiguration InitializationConfig = new InitializationConfiguration();

        /// <summary>Affinities (neighbourhood) configuration.</summary>
        public AffinitiesConfiguration AffinitiesConfig = new AffinitiesConfiguration();

        /// <summary>Locality Sensitive Hashing (LSH) Forest configuration.</summary>
        public LSHFConfiguration LSHFConfig = new LSHFConfiguration();

        /// <summary>Gradient descend configuration.</summary>
        public GradientConfiguration GradientConfig
        {
            get { return gradient.GradientConfig; }
            set { gradient.GradientConfig = value; }
        }

        /// <summary>Barnes-Hut approximation configuration.</summary>
        public BarnesHutConfiguration BarnesHutConfig
        {
            get { return gradient.BarnesHutConfig; }
            set { gradient.BarnesHutConfig = value; }
        }

        /// <summary>Polynomial Interpolation configuration.</summary>
        public PIConfiguration PIConfig
        {
            get { return gradient.PIConfig; }
            set { gradient.PIConfig = value; }
        }

        /// <summary>tSNE constructor - sets data for dimensionality reduction</summary>
        /// <param name="Data">2d array of instances (iterated by first index)</param>
        public tSNE(float[][] Data)
        {
            N = Data.Length;
            D = Data[0].Length;

            data = Data;
            Normalize(data, N, D);
        }

        private static void Normalize(float[][] array, int n, int d)
        {
            float gmin = float.MaxValue;
            float gmax = float.MinValue;
            float[] means = new float[d];

            //find min max and mean
            Parallel.For(0, d, j =>
            {
                double mean = 0;
                for (int i = 0; i < n; i++)
                {
                    float val = array[i][j];
                    if (gmin > val) gmin = val;
                    if (gmax < val) gmax = val;
                    mean += val;
                }
                means[j] = (float)(mean / n);
            });
            gmax -= gmin; //max adjusted for min, means not adjusted at all

            //normalize
            Parallel.For(0, d, j =>
            {
                float mean = means[j];
                for (int i = 0; i < n; i++) array[i][j] = (array[i][j] - mean) / gmax;
            });
        }

        /// <summary>Reduces dimensionality to dimensions using Hybrid tSNE.</summary>
        /// <param name="dimensions">Dimensionality of output.</param>
        /// <returns>2d array of reduced dimensionality instances (iterated by first index).</returns>
        public double[][] Reduce(int dimensions, bool verbose = false)
        {
            this.verbose = verbose;

            if (N < AffinitiesConfig.neighbours)
                throw new InvalidOperationException("Tried to run t-SNE with more neighbours than instances! Change perplexity or take larger data sample.");
            if (dimensions < 1)
                throw new ArgumentOutOfRangeException("Unable to reduce do less than 1 dimension! Provide valid target dimensionality.");

            Dims = dimensions;

            LSHForest.SymmetricANN(data, AffinitiesConfig.neighbours, LSHFConfig, out List<int>[] ids, out List<double>[] dists, verbose);
            data = null;

            double[][] P = HDaffinities(ids, dists);
            dists = null;

            double[] Y = InitializationConfig.SmartInit ? InitialSolution(ids, P) : InitialSolution();

            Y = gradient.GradientDescend(N, Dims, ids, P, Y, verbose);
            P = null;

            return PrepareResult(Y);
        }

        private double[] InitialSolution(List<int>[] ids = null, double[][] P = null)
        {
            //Allocate for solution and generate initial positions
            double[] Y = new double[N * Dims];
            Random R = InitializationConfig.InitialSolutionSeed == -1 ? new Random() : new Random(InitializationConfig.InitialSolutionSeed);
            for (int i = N * Dims / 2 - 1; i >= 0; i--)
            {
                double a, b;
                a = R.NextDouble();
                b = R.NextDouble();
                Y[2 * i] = Math.Sqrt(-2.0 * Math.Log(a)) * Math.Cos(2.0 * Math.PI * b); //Box-Muller transform
                Y[2 * i + 1] = Math.Sqrt(-2.0 * Math.Log(a)) * Math.Sin(2.0 * Math.PI * b); //Box-Muller transform
            }

            if (P != null)
            {
                double sigma = Math.Pow(N, 1.0 / Dims);
                for (int i = 0; i < N; i++)
                {
                    double[] avg = new double[Dims];
                    double wei = 0;
                    int count = 0;

                    for (int j = P[i].Length - 1; j >= 0; j--)
                    {
                        if (ids[i][j] < i)
                        {
                            for (int k = 0; k < Dims; k++) avg[k] += Y[ids[i][j] * Dims + k] * P[i][j];
                            wei += P[i][j];
                            count++;
                        }
                    }

                    if (count > 0)
                        for (int k = 0; k < Dims; k++)
                        {
                            Y[i * Dims + k] /= sigma;
                            Y[i * Dims + k] += avg[k] / wei;
                        }
                }
            }

            return Y;
        }

        private double[][] PrepareResult(double[] Y)
        {
            double[][] res = new double[N][];
            Parallel.For(0, N, i =>
            {
                res[i] = new double[Dims];
                for (int j = 0; j < Dims; j++) res[i][j] = Y[i * Dims + j];
            });
            return res;
        }

        private double[][] HDaffinities(List<int>[] neiIds, List<double>[] neiDists)
        {
            double[][] P = new double[N][]; //probability matrix
            for (int i = 0; i < N; i++) P[i] = new double[neiDists[i].Count];

            double[] betas = new double[N]; //precision vector
            for (int i = 0; i < N; i++) betas[i] = 1.0;

            Parallel.For(0, N, i =>
            {
                double betamin = 0;
                double betamax = double.PositiveInfinity;

                GaussKernel(neiDists[i], betas[i], out double currEntropy, out double sumP, P[i]);
                double EntropyDiff = currEntropy - AffinitiesConfig.entropy;

                for (int j = 0; Math.Abs(EntropyDiff) > AffinitiesConfig.EntropyTol && j < AffinitiesConfig.EntropyIter; j++)
                {
                    if (EntropyDiff > 0)
                    {
                        betamin = betas[i];
                        betas[i] = (double.IsInfinity(betamax)) ? betas[i] * 2.0 : (betamin + betamax) / 2.0;
                    }
                    else
                    {
                        betamax = betas[i];
                        betas[i] = (double.IsInfinity(betamin)) ? betas[i] / 2.0 : (betamin + betamax) / 2.0;
                    }

                    GaussKernel(neiDists[i], betas[i], out currEntropy, out sumP, P[i]);
                    EntropyDiff = currEntropy - AffinitiesConfig.entropy;
                }

                for (int j = neiDists[i].Count - 1; j >= 0; --j) P[i][j] /= sumP;
            });

            Parallel.For(0, N, i =>
            {
                for (int j = P[i].Length - 1; j >= 0; j--)
                {
                    if (i < neiIds[i][j])
                    {
                        int id = neiIds[i][j];

                        int jd = 0;
                        while (neiIds[id][jd] != i) jd++;

                        double val = 0.5 * (P[i][j] + P[id][jd]) / N;
                        P[i][j] = val;
                        P[id][jd] = val;
                    }
                }
            });

            return P;
        }

        private void GaussKernel(List<double> neiDists, double beta, out double currEntropy, out double sumP, double[] Pi)
        {
            sumP = 0;
            double sumDP = 0;
            int nei = neiDists.Count;
            for (int j = 0; j < nei; j++) Pi[j] = Math.Exp(-neiDists[j] * beta);
            for (int j = 0; j < nei; j++)
            {
                sumP += Pi[j];
                sumDP += neiDists[j] * Pi[j];
            }
            currEntropy = Math.Log(sumP, 2) + beta * sumDP / sumP;
        }
    }
}