using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;

namespace Hybrid_tSNE
{
    internal class Gradient
    {
        private int N;
        private int Dims;
        private List<int>[] ids;
        private double[][] P;
        private double[] Y;

        private double[] Ydelta;
        private double[] gains;
        private double[] Grad;
        private double[] GradT;

        private HashSpatialTree HST;
        private FFTRepulsion FFTR;

        private const int repulsionTries = 5;
        /// <summary>Used repulsion method</summary>
        public RepulsionMethods ChosenRepulsionMethod { get; private set; }

        /// <summary>Gradient descend configuration.</summary>
        public GradientConfiguration GradientConfig = new GradientConfiguration();

        /// <summary>Barnes-Hut approximation configuration.</summary>
        public BarnesHutConfiguration BarnesHutConfig = new BarnesHutConfiguration();

        /// <summary>Polynomial Interpolation configuration.</summary>
        public PIConfiguration PIConfig = new PIConfiguration();

        private void Initialize(int N, int Dims, List<int>[] ids, double[][] P, double[] Y)
        {
            this.N = N;
            this.Dims = Dims;
            this.ids = ids;
            this.P = P;
            this.Y = Y;

            // initialize reusable allocations
            Grad = new double[N * Dims];
            GradT = new double[N * Dims];
            Ydelta = new double[N * Dims];
            gains = new double[N * Dims];
            for (int i = N * Dims - 1; i >= 0; i--) gains[i] = 1;

            //initialize repulsion methods
            ChosenRepulsionMethod = GradientConfig.RepulsionMethod;
            if (ChosenRepulsionMethod == RepulsionMethods.auto || ChosenRepulsionMethod == RepulsionMethods.barnes_hut) HST = new HashSpatialTree(N, Dims, Y);
            if (ChosenRepulsionMethod == RepulsionMethods.auto || ChosenRepulsionMethod == RepulsionMethods.fft) InitializeFFTRepulsion(Y, GradT);
        }

        public double[] GradientDescend(int N, int Dims, List<int>[] ids, double[][] P, double[] Y, bool verbose = false)
        {
            Initialize(N, Dims, ids, P, Y);

            //initialize repulsion method test
            int num_methods = Enum.GetValues(typeof(RepulsionMethods)).Length;
            long[] test_repulsion = new long[num_methods];
            test_repulsion[0] = 1;
            for (int i = 1; i < num_methods; i++) test_repulsion[i] = long.MaxValue;
            int tries = 0;

            //main loop
            for (int t = 0; t < GradientConfig.Iterations; t++)
            {
                if (verbose && t % 50 == 0) Console.WriteLine("Gradient {0}% complete", 100 * (t + 1) / GradientConfig.Iterations);

                if (ChosenRepulsionMethod != RepulsionMethods.auto) Grad = GradientStep(GradientConfig.Exaggeration(t, GradientConfig.Iterations), ChosenRepulsionMethod);
                else
                {
                    if ((RepulsionMethods)test_repulsion[0] == RepulsionMethods.fft && FFTR == null) //FFTRepulsion unavailable for given Dims
                    {
                        tries = repulsionTries;
                        t--;
                    }
                    else
                    {
                        Stopwatch gradient = Stopwatch.StartNew();
                        Grad = GradientStep(GradientConfig.Exaggeration(t, GradientConfig.Iterations), (RepulsionMethods)test_repulsion[0]);
                        gradient.Stop();
                        if (test_repulsion[test_repulsion[0]] > gradient.ElapsedMilliseconds) test_repulsion[test_repulsion[0]] = gradient.ElapsedMilliseconds;
                        tries++;
                    }

                    if (tries == repulsionTries)
                    {
                        tries = 0;
                        test_repulsion[0]++;

                        if (test_repulsion[0] == num_methods) ChooseRepulsionMethod(test_repulsion);
                    }
                }

                //calculate next Y
                double alpha = GradientConfig.Momentum(t, GradientConfig.Iterations);
                double eta = GradientConfig.LearningRate(t, GradientConfig.Iterations);
                for (int i = N * Dims - 1; i >= 0; i--)
                {
                    if (Math.Sign(Grad[i]) == Math.Sign(Ydelta[i])) gains[i] += 0.2;
                    else
                    {
                        gains[i] *= 0.8;
                        if (gains[i] < GradientConfig.GradMinGain) gains[i] = GradientConfig.GradMinGain;
                    }

                    double val = alpha * Ydelta[i] + eta * gains[i] * Grad[i];
                    Ydelta[i] = val;
                    Y[i] += val;
                }
            }

            return Y;
        }

        private void InitializeFFTRepulsion(double[] Y, double[] GradT)
        {
            switch (Dims)
            {
                case 1:
                    FFTR = new FFTRepulsion1D(Y, GradT, N, PIConfig);
                    break;
                case 2:
                    FFTR = new FFTRepulsion2D(Y, GradT, N, PIConfig);
                    break;
                case 3:
                    FFTR = new FFTRepulsion3D(Y, GradT, N, PIConfig);
                    break;
                default:
                    switch (ChosenRepulsionMethod)
                    {
                        case RepulsionMethods.fft:
                            throw new NotImplementedException("Polynomial interpolation for dimensionality larger than 3D is not implemented!");
                        case RepulsionMethods.auto:
                            Console.WriteLine("Polynomial interpolation for dimensionality larger than 3D is not implemented!");
                            break;
                    }
                    break;
            }
        }

        private void ChooseRepulsionMethod(long[] test_repulsion)
        {
            long min_time = long.MaxValue;
            for (int i = 1; i < test_repulsion.Length; i++)
                if (min_time > test_repulsion[i])
                {
                    min_time = test_repulsion[i];
                    ChosenRepulsionMethod = (RepulsionMethods)i;
                }

            Console.WriteLine("Chosen repulsion method:\t{0}", ChosenRepulsionMethod);

            if (ChosenRepulsionMethod != RepulsionMethods.barnes_hut) HST = null;
            if (ChosenRepulsionMethod != RepulsionMethods.fft) FFTR = null;
        }

        private double[] GradientStep(double exagg, RepulsionMethods method)
        {
            Task ATask = Task.Run(() =>
            {
                Grad = GradientAttractive(Grad, exagg);
            });

            double Z = 0;
            Task RTask = Task.Run(() =>
            {
                switch (method)
                {
                    case RepulsionMethods.barnes_hut:
                        HST.RebuildTree(BarnesHutConfig.Presort);
                        Array.Clear(GradT, 0, GradT.Length);
                        Z = HST.EstimateRepulsion(GradT, BarnesHutConfig.theta2);
                        break;
                    case RepulsionMethods.fft:
                        FFTR.ComputeFFTGradient();
                        break;
                }
            });

            Task.WaitAll(new Task[] { RTask, ATask });

            switch (method)
            {
                case RepulsionMethods.barnes_hut:
                    Z = 1.0 / Z;
                    for (int i = N * Dims - 1; i >= 0; i--) Grad[i] += GradT[i] * Z;
                    break;
                case RepulsionMethods.fft:
                    for (int i = N * Dims - 1; i >= 0; i--) Grad[i] += GradT[i];
                    break;
            }

            return Grad;
        }

        private double[] GradientAttractive(double[] GradA, double exagg)
        {
            Parallel.For(0, N, i =>
            {
                double diff;
                double weight;
                double[] dim = new double[Dims];
                int yi = i * Dims;

                for (int j = P[i].Length - 1; j >= 0; j--)
                {
                    int yj = ids[i][j] * Dims;
                    weight = 0;
                    for (int k = 0; k < Dims; k++)
                    {
                        diff = Y[yi + k] - Y[yj + k];
                        weight += diff * diff;
                    }
                    weight = P[i][j] / (1 + weight);

                    for (int k = 0; k < Dims; k++) dim[k] += weight * (Y[yi + k] - Y[yj + k]);
                }

                for (int k = 0; k < Dims; k++) GradA[yi + k] = -exagg * dim[k];
            });

            return GradA;
        }
    }
}
