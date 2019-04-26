using System;
using System.Linq;
using System.Threading.Tasks;
using System.Numerics;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;

namespace Hybrid_tSNE
{
    abstract class FFTRepulsion
    {
        //constants
        internal static readonly int[] allowed_n_boxes_per_dim = new int[] { 25, 36, 50, 55, 60, 65, 70, 75, 80, 85, 90, 96, 100, 110, 120, 130, 140, 150, 175, 200 };
        internal readonly int n_terms;

        //data
        internal double[] Y;
        internal double[] grad;
        internal int N;
        internal int D;

        //parameters
        internal int n_interpolation_points;
        internal double min_num_intervals;
        internal double intervals_per_integer;

        //helpers
        internal double min, max;
        internal int n_boxes;
        internal int n_boxes_per_dim;
        internal double box_width;
        internal int n_interpolation_points_1d;
        internal double h;

        //allocations
        internal double[] box_bounds;
        internal double[] kernel_tilde;
        internal double[] tilde;
        internal double[] tilde_values;
        internal readonly int[] bins;
        internal readonly double[] in_box;
        internal readonly double[] interpolated_values;
        internal readonly double[] interpolation_denominators;
        internal readonly double[] potentialsQij;


        public FFTRepulsion(double[] Y, double[] GradR, int N, int D, PIConfiguration PIConfig)
        {
            Control.UseNativeMKL();  //required by FFTRepulsion

            this.Y = Y;
            grad = GradR;
            this.N = N;
            this.D = D;
            n_terms = D + 2;
            n_interpolation_points = PIConfig.n_interpolation_points;
            h = 1.0 / n_interpolation_points;
            min_num_intervals = PIConfig.min_num_intervals;
            intervals_per_integer = PIConfig.intervals_per_integer;

            bins = new int[N * D];
            in_box = new double[N * D];
            interpolated_values = new double[N * D * n_interpolation_points];
            interpolation_denominators = new double[n_interpolation_points];
            CalculateDenominators();
            potentialsQij = new double[N * n_terms];
        }

        internal abstract void CalculatePotentials();

        internal abstract void Clear();

        internal abstract void Reallocate();

        public void ComputeFFTGradient()
        {
            UpdateMinMax();
            SetupBoxes();
            BinPoints();
            Interpolate();
            CalculatePotentials();

            // Compute the normalization constant Z or sum of q_{ij}. This expression is different from the one in the original
            // paper, but equivalent. This is done so we need only use a single kernel (K_2 in the paper) instead of two
            // different ones. We subtract N at the end because the following sums over all i, j, whereas Z contains i \neq j
            double sum_Q = 0;
            for (int i = 0; i < N; i++)
            {
                double sqrsum = 0;
                for (int k = 0; k < D; k++) sqrsum += Y[D * i + k] * Y[D * i + k];

                double sum = 0;
                for (int k = 0; k < D; k++) sum += Y[D * i + k] * potentialsQij[i * n_terms + k + 1];

                sum_Q += (1 + sqrsum) * potentialsQij[i * n_terms] - 2 * sum + potentialsQij[(i + 1) * n_terms - 1];
            }
            sum_Q -= N;

            // Make the negative term, or F_rep in the equation 3 of the paper
            for (int i = 0; i < N; i++)
                for (int j = 0; j < D; j++)
                    grad[D * i + j] = (Y[D * i + j] * potentialsQij[i * n_terms] - potentialsQij[i * n_terms + j + 1]) / sum_Q;
        }

        private void CalculateDenominators()
        {
            for (int i = 0; i < n_interpolation_points; i++)
            {
                interpolation_denominators[i] = 1;
                for (int j = 0; j < n_interpolation_points; j++)
                    if (i != j)
                        interpolation_denominators[i] *= (i - j) * h;
            }
        }

        private void UpdateMinMax()
        {
            min = double.PositiveInfinity;
            max = double.NegativeInfinity;
            for (int i = N * D - 1; i >= 0; i--)
            {
                if (Y[i] < min) min = Y[i];
                if (Y[i] > max) max = Y[i];
            }
        }

        private int GetValidNBox()
        {
            // Compute the number of boxes in a single dimension and the total number of boxes in 2d
            int n_boxes_per_dim = (int)Math.Max(min_num_intervals, (max - min) / intervals_per_integer);

            // MKL and FFTW works faster on radix 2, 3, 5, 7, 11, 13
            if (n_boxes_per_dim < allowed_n_boxes_per_dim.Last())
            {
                int i;
                for (i = 0; allowed_n_boxes_per_dim[i] < n_boxes_per_dim; i++) ;
                n_boxes_per_dim = allowed_n_boxes_per_dim[i];
            }

            return n_boxes_per_dim;
        }

        private void SetupBoxes()
        {
            int n_boxes_per_dim = GetValidNBox();

            if (n_boxes_per_dim != this.n_boxes_per_dim)
            {
                this.n_boxes_per_dim = n_boxes_per_dim;
                n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim;
                n_boxes = (int)Math.Pow(n_boxes_per_dim, D);

                box_bounds = new double[n_boxes_per_dim + 1];
                tilde = new double[n_interpolation_points_1d];
                tilde_values = new double[(int)Math.Pow(n_interpolation_points_1d, D) * n_terms];
                kernel_tilde = new double[(int)Math.Pow(2 * n_interpolation_points_1d + (D == 1 ? 2 : 0), D)];
                Reallocate();
            }
            else
            {
                Array.Clear(tilde_values, 0, tilde_values.Length);
		Clear();
            }

            box_width = (max - min) / n_boxes_per_dim;
            for (int i = 0; i <= n_boxes_per_dim; i++) box_bounds[i] = i * box_width + min;

            // Coordinates of all the equispaced interpolation points
            double h = box_width / n_interpolation_points;
            tilde[0] = min + h / 2;
            for (int i = 1; i < n_interpolation_points_1d; i++) tilde[i] = tilde[i - 1] + h;
        }

        private void BinPoints() // Determine which box each point belongs to and the relative position of each point in its box in the interval [0, 1]
        {
            for (int i = 0; i < N; i++)
                for (int k = 0; k < D; k++)
                {
                    int id = D * i + k;
                    in_box[id] = (Y[id] - min) / box_width;
                    bins[id] = (int)in_box[id];
                    in_box[id] -= bins[id];

                    if (bins[id] == n_boxes_per_dim)
                    {
                        bins[id]--;
                        in_box[id] = 1;
                    }
                }
        }

        private void Interpolate()
        {
            // Compute the numerators and the interpolant value
            Parallel.For(0, N, i =>
            {
                for (int k = 0; k < D; k++)
                    for (int j = 0; j < n_interpolation_points; j++)
                    {
                        int id = D * (j * N + i) + k;
                        double value = 1;
                        for (int l = 0; l < n_interpolation_points; l++)
                            if (j != l)
                                value *= in_box[D * i + k] - (0.5 + l) * h;
                        interpolated_values[id] = value / interpolation_denominators[j];
                    }
            });
        }
    }

    class FFTRepulsion1D : FFTRepulsion
    {
        internal double[] fft;

        public FFTRepulsion1D(double[] Y, double[] GradR, int N, PIConfiguration PIConfig) : base(Y, GradR, N, 1, PIConfig) { }

        internal override void CalculatePotentials()
        {
            ComputeW();
            ComputeKernelTilde();
            NBodyFFT();
            ComputePotentials();
        }

        internal override void Reallocate()
        {
            fft = new double[2 * n_interpolation_points_1d + 2];
        }

        internal override void Clear()
        {
        }

        private void ComputeW()
        {
            Parallel.For(0, N, i =>
            {
                int box_i = bins[i] * n_interpolation_points;
                double sqrsum = Y[i] * Y[i];

                for (int interp_i = 0; interp_i < n_interpolation_points; interp_i++)
                {
                    // Compute the index of the point in the interpolation grid of points
                    int id = n_terms * (box_i + interp_i);
                    double product = interpolated_values[interp_i * N + i];
                    tilde_values[id] += product;
                    tilde_values[id + 1] += product * Y[i];
                    tilde_values[id + 2] += product * sqrsum;
                }
            });
        }

        private void ComputeKernelTilde()
        {
            //Evaluate the kernel at the interpolation nodes and form the embedded generating kernel vector for a circulant matrix
            for (int i = 0; i < n_interpolation_points_1d; i++)
                    kernel_tilde[n_interpolation_points_1d + i]
                    = kernel_tilde[n_interpolation_points_1d - i]
                    = SquaredCauchy(tilde[0], tilde[i]);

            // Precompute the FFT of the kernel generating matrix
            Fourier.ForwardReal(kernel_tilde, 2 * n_interpolation_points_1d, FourierOptions.NoScaling);
        }

        double SquaredCauchy(double x, double y)
        {
            double tmp = x - y;
            tmp = 1 + tmp * tmp;
            return 1 / (tmp * tmp);
        }

        private void NBodyFFT()
        {
            //Compute the values v_{m, n} at the equispaced nodes, multiply the kernel matrix with the coefficients w
            int n_fft_coeffs = 2 * n_interpolation_points_1d;
            
            for (int d = 0; d < n_terms; d++)
            {
                Array.Clear(fft, 0, n_interpolation_points_1d);
                for (int i = 0; i < n_interpolation_points_1d; i++) fft[n_interpolation_points_1d + i] = tilde_values[i * n_terms + d];
                Array.Clear(fft, n_fft_coeffs, fft.Length - n_fft_coeffs);

                Fourier.ForwardReal(fft, n_fft_coeffs);

                // Take the Hadamard product of two complex vectors (kernel tilde is real)
                for (int i = n_interpolation_points_1d; i >= 0; i--)
                {
                    fft[2 * i] *= kernel_tilde[2 * i];
                    fft[2 * i + 1] *= kernel_tilde[2 * i];
                }

                // Invert the computed values at the interpolated nodes
                Fourier.InverseReal(fft, n_fft_coeffs);

                for (int i = 0; i < n_interpolation_points_1d; i++)
                    tilde_values[i * n_terms + d] = fft[i];
            }
        }
        
        private void ComputePotentials() //Compute the potentials \tilde{\phi}
        {
            Array.Clear(potentialsQij, 0, potentialsQij.Length);
            Parallel.For(0, N, i =>
            {
                int box_i = bins[i] * n_interpolation_points;
                int id = i * n_terms;

                for (int interp_i = 0; interp_i < n_interpolation_points; interp_i++)
                {
                    // Compute the index of the point in the interpolation grid of points
                    int idx = n_terms * (box_i + interp_i);
                    double interpolated = interpolated_values[interp_i * N + i];

                    potentialsQij[id] += interpolated * tilde_values[idx];
                    potentialsQij[id + 1] += interpolated * tilde_values[idx + 1];
                    potentialsQij[id + 2] += interpolated * tilde_values[idx + 2];
                }
            });
        }
    }

    class FFTRepulsion2D : FFTRepulsion
    {
        internal Complex[] fft;

        public FFTRepulsion2D(double[] Y, double[] GradR, int N, PIConfiguration PIConfig) : base(Y, GradR, N, 2, PIConfig) { }

        internal override void CalculatePotentials()
        {
            ComputeW();
            ComputeKernelTilde();
            NBodyFFT();
            ComputePotentials();
        }

        internal override void Reallocate()
        {
		fft = new Complex[(int)Math.Pow(2 * n_interpolation_points_1d, 2)];
        }

        internal override void Clear()
        {
            Array.Clear(fft, 0, fft.Length);
        }

        private void ComputeW()
        {
            Parallel.For(0, N, i =>
            {
                int box_i = bins[i * 2] * n_interpolation_points;
                int box_j = bins[i * 2 + 1] * n_interpolation_points;
                double sqrsum = Y[2 * i] * Y[2 * i] + Y[2 * i + 1] * Y[2 * i + 1];

                for (int interp_i = 0; interp_i < n_interpolation_points; interp_i++)
                    for (int interp_j = 0; interp_j < n_interpolation_points; interp_j++)
                    {
                        // Compute the index of the point in the interpolation grid of points
                        int id = n_terms * ((box_i + interp_i) * n_interpolation_points_1d + box_j + interp_j);
                        double product = interpolated_values[2 * (interp_i * N + i)] * interpolated_values[2 * (interp_j * N + i) + 1];
                        tilde_values[id] += product;
                        tilde_values[id + 1] += product * Y[2 * i];
                        tilde_values[id + 2] += product * Y[2 * i + 1];
                        tilde_values[id + 3] += product * sqrsum;
                    }
            });
        }

        private void ComputeKernelTilde()
        {
            //Evaluate the kernel at the interpolation nodes and form the embedded generating kernel vector for a circulant matrix
            int n_fft_coeffs = 2 * n_interpolation_points_1d;
            for (int i = 0; i < n_interpolation_points_1d; i++)
                for (int j = 0; j < n_interpolation_points_1d; j++)
                    fft[(n_interpolation_points_1d + i) * n_fft_coeffs + n_interpolation_points_1d + j]
                    = fft[(n_interpolation_points_1d - i) * n_fft_coeffs + n_interpolation_points_1d + j]
                    = fft[(n_interpolation_points_1d + i) * n_fft_coeffs + n_interpolation_points_1d - j]
                    = fft[(n_interpolation_points_1d - i) * n_fft_coeffs + n_interpolation_points_1d - j]
                    = SquaredCauchy(tilde[0], tilde[i], tilde[j]);

            // Precompute the FFT of the kernel generating matrix
            Fourier.Forward2D(fft, n_fft_coeffs, n_fft_coeffs, FourierOptions.NoScaling);
            for (int i = fft.Length - 1; i >= 0; i--) kernel_tilde[i] = fft[i].Real;
        }

        static private double SquaredCauchy(double x, double y1, double y2)
        {
            double diff = x - y1;
            double tmp = 1 + diff * diff;
            diff = x - y2;
            tmp += diff * diff;
            return 1 / (tmp * tmp);
        }

        private void NBodyFFT()
        {
            //Compute the values v_{m, n} at the equispaced nodes, multiply the kernel matrix with the coefficients w
            int n_fft_coeffs = 2 * n_interpolation_points_1d;

            for (int d = 0; d < n_terms; d += 2)
            {
                Array.Clear(fft, 0, fft.Length);
                for (int i = 0; i < n_interpolation_points_1d; i++)
                    for (int j = 0; j < n_interpolation_points_1d; j++)
                        fft[i * n_fft_coeffs + j] = new Complex(tilde_values[(i * n_interpolation_points_1d + j) * n_terms + d], tilde_values[(i * n_interpolation_points_1d + j) * n_terms + d + 1]);

                Fourier.Forward2D(fft, n_fft_coeffs, n_fft_coeffs);

                // Take the Hadamard product of two complex vectors
                for (int i = kernel_tilde.Length - 1; i >= 0; i--)
                    fft[i] *= kernel_tilde[i];

                // Invert the computed values at the interpolated nodes
                Fourier.Inverse2D(fft, n_fft_coeffs, n_fft_coeffs);

                for (int i = 0; i < n_interpolation_points_1d; i++)
                    for (int j = 0; j < n_interpolation_points_1d; j++)
                    {
                        tilde_values[(i * n_interpolation_points_1d + j) * n_terms + d] = fft[(n_interpolation_points_1d + i) * n_fft_coeffs + n_interpolation_points_1d + j].Real;
                        tilde_values[(i * n_interpolation_points_1d + j) * n_terms + d + 1] = fft[(n_interpolation_points_1d + i) * n_fft_coeffs + n_interpolation_points_1d + j].Imaginary;
                    }
            }
        }
        
        private void ComputePotentials() //Compute the potentials \tilde{\phi}
        {
            Array.Clear(potentialsQij, 0, potentialsQij.Length);
            Parallel.For(0, N, i =>
            {
                int box_i = bins[i * 2] * n_interpolation_points;
                int box_j = bins[i * 2 + 1] * n_interpolation_points;
                int id = i * n_terms;

                for (int interp_i = 0; interp_i < n_interpolation_points; interp_i++)
                    for (int interp_j = 0; interp_j < n_interpolation_points; interp_j++)
                    {
                        // Compute the index of the point in the interpolation grid of points
                        int idx = n_terms * ((box_i + interp_i) * n_interpolation_points_1d + box_j + interp_j);
                        double interpolated = interpolated_values[2 * (interp_i * N + i)] * interpolated_values[2 * (interp_j * N + i) + 1];

                        potentialsQij[id + 0] += interpolated * tilde_values[idx + 0];
                        potentialsQij[id + 1] += interpolated * tilde_values[idx + 1];
                        potentialsQij[id + 2] += interpolated * tilde_values[idx + 2];
                        potentialsQij[id + 3] += interpolated * tilde_values[idx + 3];
                    }
            });
        }
    }

    class FFTRepulsion3D : FFTRepulsion
    {
        internal Complex[] fft;

        public FFTRepulsion3D(double[] Y, double[] GradR, int N, PIConfiguration PIConfig) : base(Y, GradR, N, 3, PIConfig) { }

        internal override void CalculatePotentials()
        {
            ComputeW();
            ComputeKernelTilde();
            NBodyFFT();
            ComputePotentials();
        }

        internal override void Reallocate()
        {
            fft = new Complex[(int)Math.Pow(2 * n_interpolation_points_1d, 3)];
        }

        internal override void Clear()
        {
            Array.Clear(fft, 0, fft.Length);
        }

        private void ComputeW()
        {
            Parallel.For(0, N, i =>
            {
                int box_i = bins[i * 3] * n_interpolation_points;
                int box_j = bins[i * 3 + 1] * n_interpolation_points;
                int box_k = bins[i * 3 + 2] * n_interpolation_points;
                double sqrsum = Y[3 * i] * Y[3 * i] + Y[3 * i + 1] * Y[3 * i + 1] + Y[3 * i + 2] * Y[3 * i + 2];

                for (int interp_i = 0; interp_i < n_interpolation_points; interp_i++)
                    for (int interp_j = 0; interp_j < n_interpolation_points; interp_j++)
                        for (int interp_k = 0; interp_k < n_interpolation_points; interp_k++)
                        {
                            // Compute the index of the point in the interpolation grid of points
                            int id = n_terms * (((box_i + interp_i) * n_interpolation_points_1d + box_j + interp_j) * n_interpolation_points_1d + box_k + interp_k);
                            double product = interpolated_values[3 * (interp_i * N + i)] * interpolated_values[3 * (interp_j * N + i) + 1] * interpolated_values[3 * (interp_k * N + i) + 2];
                            tilde_values[id] += product;
                            tilde_values[id + 1] += product * Y[3 * i];
                            tilde_values[id + 2] += product * Y[3 * i + 1];
                            tilde_values[id + 3] += product * Y[3 * i + 2];
                            tilde_values[id + 4] += product * sqrsum;
                        }
            });
        }

        private void ComputeKernelTilde()
        {
            //Evaluate the kernel at the interpolation nodes and form the embedded generating kernel vector for a circulant matrix
            int n_fft_coeffs = 2 * n_interpolation_points_1d;
            for (int i = 0; i < n_interpolation_points_1d; i++)
                for (int j = 0; j < n_interpolation_points_1d; j++)
                    for (int k = 0; k < n_interpolation_points_1d; k++)
                        fft[((n_interpolation_points_1d + i) * n_fft_coeffs + n_interpolation_points_1d + j) * n_fft_coeffs + n_interpolation_points_1d + k]
                        = fft[((n_interpolation_points_1d - i) * n_fft_coeffs + n_interpolation_points_1d + j) * n_fft_coeffs + n_interpolation_points_1d + k]
                        = fft[((n_interpolation_points_1d + i) * n_fft_coeffs + n_interpolation_points_1d - j) * n_fft_coeffs + n_interpolation_points_1d + k]
                        = fft[((n_interpolation_points_1d - i) * n_fft_coeffs + n_interpolation_points_1d - j) * n_fft_coeffs + n_interpolation_points_1d + k]
                        = fft[((n_interpolation_points_1d + i) * n_fft_coeffs + n_interpolation_points_1d + j) * n_fft_coeffs + n_interpolation_points_1d - k]
                        = fft[((n_interpolation_points_1d - i) * n_fft_coeffs + n_interpolation_points_1d + j) * n_fft_coeffs + n_interpolation_points_1d - k]
                        = fft[((n_interpolation_points_1d + i) * n_fft_coeffs + n_interpolation_points_1d - j) * n_fft_coeffs + n_interpolation_points_1d - k]
                        = fft[((n_interpolation_points_1d - i) * n_fft_coeffs + n_interpolation_points_1d - j) * n_fft_coeffs + n_interpolation_points_1d - k]
                        = SquaredCauchy(tilde[0], tilde[i], tilde[j], tilde[k]);

            // Precompute the FFT of the kernel generating matrix
            Fourier.ForwardMultiDim(fft, new int[] { n_fft_coeffs, n_fft_coeffs, n_fft_coeffs }, FourierOptions.NoScaling);
            for (int i = fft.Length - 1; i >= 0; i--) kernel_tilde[i] = fft[i].Real;
        }

        static private double SquaredCauchy(double x, double y1, double y2, double y3)
        {
            double diff = x - y1;
            double tmp = 1 + diff * diff;
            diff = x - y2;
            tmp += diff * diff;
            diff = x - y3;
            tmp += diff * diff;
            return 1 / (tmp * tmp);
        }

        private void NBodyFFT()
        {
            //Compute the values v_{m, n} at the equispaced nodes, multiply the kernel matrix with the coefficients w
            int n_fft_coeffs = 2 * n_interpolation_points_1d;

            for (int d = 0; d < n_terms; d += 2)
            {
                Array.Clear(fft, 0, fft.Length);
                for (int i = 0; i < n_interpolation_points_1d; i++)
                    for (int j = 0; j < n_interpolation_points_1d; j++)
                        for (int k = 0; k < n_interpolation_points_1d; k++)
                            fft[(i * n_fft_coeffs + j) * n_fft_coeffs + k] = new Complex(tilde_values[((i * n_interpolation_points_1d + j) * n_interpolation_points_1d + k) * n_terms + d],
                                (d == n_terms - 1) ? 0 : tilde_values[((i * n_interpolation_points_1d + j) * n_interpolation_points_1d + k) * n_terms + d + 1]);

                Fourier.ForwardMultiDim(fft, new int[] { n_fft_coeffs, n_fft_coeffs, n_fft_coeffs });

                // Take the Hadamard product of two complex vectors
                for (int i = kernel_tilde.Length - 1; i >= 0; i--)
                    fft[i] *= kernel_tilde[i];

                // Invert the computed values at the interpolated nodes
                Fourier.InverseMultiDim(fft, new int[] { n_fft_coeffs, n_fft_coeffs, n_fft_coeffs });

                for (int i = 0; i < n_interpolation_points_1d; i++)
                    for (int j = 0; j < n_interpolation_points_1d; j++)
                        for (int k = 0; k < n_interpolation_points_1d; k++)
                        {
                            tilde_values[((i * n_interpolation_points_1d + j) * n_interpolation_points_1d + k) * n_terms + d] = fft[((n_interpolation_points_1d + i) * n_fft_coeffs + n_interpolation_points_1d + j) * n_fft_coeffs + n_interpolation_points_1d + k].Real;
                            if (d != n_terms - 1)
                                tilde_values[((i * n_interpolation_points_1d + j) * n_interpolation_points_1d + k) * n_terms + d + 1] = fft[((n_interpolation_points_1d + i) * n_fft_coeffs + n_interpolation_points_1d + j) * n_fft_coeffs + n_interpolation_points_1d + k].Imaginary;
                        }
            }
        }
        
        private void ComputePotentials() //Compute the potentials \tilde{\phi}
        {
            Array.Clear(potentialsQij, 0, potentialsQij.Length);
            Parallel.For(0, N, i =>
            {
                int box_i = bins[i * 3] * n_interpolation_points;
                int box_j = bins[i * 3 + 1] * n_interpolation_points;
                int box_k = bins[i * 3 + 2] * n_interpolation_points;
                int id = i * n_terms;

                for (int interp_i = 0; interp_i < n_interpolation_points; interp_i++)
                    for (int interp_j = 0; interp_j < n_interpolation_points; interp_j++)
                        for (int interp_k = 0; interp_k < n_interpolation_points; interp_k++)
                        {
                            // Compute the index of the point in the interpolation grid of points
                            int idx = n_terms * (((box_i + interp_i) * n_interpolation_points_1d + box_j + interp_j) * n_interpolation_points_1d + box_k + interp_k);
                            double interpolated = interpolated_values[3 * (interp_i * N + i)] * interpolated_values[3 * (interp_j * N + i) + 1] * interpolated_values[3 * (interp_k * N + i) + 2];

                            potentialsQij[id] += interpolated * tilde_values[idx];
                            potentialsQij[id + 1] += interpolated * tilde_values[idx + 1];
                            potentialsQij[id + 2] += interpolated * tilde_values[idx + 2];
                            potentialsQij[id + 3] += interpolated * tilde_values[idx + 3];
                            potentialsQij[id + 4] += interpolated * tilde_values[idx + 4];
                        }
            });
        }
    }
}
