using System;

namespace Hybrid_tSNE
{
    public class InitializationConfiguration
    {
        /// <summary>Smart initialization based on affinities.</summary>
        public bool SmartInit = true;

        /// <summary>Initial solution seed. If set to -1 uses random seed.</summary>
        public int InitialSolutionSeed = -1;
    }

    public class AffinitiesConfiguration
    {
        internal double entropy = Math.Log(30, 2); //entropy=log2(perplexity)

        internal int neighbours = 90; //3*perplexity

        /// <summary>Perplexity - measure of neighbourhood. neighbours = 3 * perplexity and entropy = log2(perplexity)</summary>
        public double Perplexity
        {
            get => Math.Exp(entropy);
            set
            {
                entropy = Math.Log(value, 2);
                neighbours = (int)(3 * value);
            }
        }

        /// <summary>Entropy tolerance for affinity search.</summary>
        public double EntropyTol = 1e-5;

        /// <summary>Max iterations for affinity search.</summary>
        public int EntropyIter = 50;
    }

    public class LSHFConfiguration
    {
        /// <summary>Number of LSH Forest trees.</summary>
        public int LSHForestTrees = 64;

        /// <summary>LSH Tree neighbours coefficient. #neighbours_per_tree = LSHTreeC * neighbours / LSHForestTrees</summary>
        public int LSHTreeC = 4;

        /// <summary>Number of LSH hash nonzero dimensions.</summary>
        public int LSHHashDims = 10;

        /// <summary>LSH Forest random generators seed. If set to -1 uses random seed.</summary>
        public int LSHSeed = -1;
    }

    public class GradientConfiguration
    {
        /// <summary>Number of gradient iterations.</summary>
        public int Iterations = 1000;

        /// <summary>Attractive force exaggeration. Gets iteration and number of iterations, returns exaggeration coefficient.</summary>
        public Func<int, int, double> Exaggeration = (t, T) => (t < 250) ? 12 - 11 / (1 + Math.Exp(-0.05 * (t - 200))) : 1;

        /// <summary>Momentum - alpha. Gets iteration and number of iterations, returns momentum.</summary>
        public Func<int, int, double> Momentum = (t, T) => (t < 250) ? 0.5 : 0.8;

        /// <summary>Learning rate - eta. Gets iteration and number of iterations, returns learning rate.</summary>
        public Func<int, int, double> LearningRate = (t, T) => 200;

        /// <summary>Set repulsion method</summary>
        public RepulsionMethods RepulsionMethod = RepulsionMethods.auto;
        
        /// <summary>Minimum allowed gradient gain value.</summary>
        public double GradMinGain = 0.01;
    }

    public class BarnesHutConfiguration
    {
        /// <summary>Barnes-Hut approximation condition squared - theta squared.</summary>
        internal double theta2 = 0.25;

        /// <summary>Barnes-Hut approximation condition - theta. Cell is used as summary if size / distance &lt; theta, size is half of cell side.</summary>
        public double BarnesHutCondition
        {
            get => Math.Sqrt(theta2);
            set => theta2 = value * value;
        }

        /// <summary>Switches tree building method between presort + find divisions and find divisions when sorting. </summary>
        public bool Presort = false;
    }

    public class PIConfiguration
    {
        /// <summary>Polynomial interpolation minimum intervals.</summary>
        public int min_num_intervals = 50;

        /// <summary>Polynomial interpolation intervals per unit of spread</summary>
        public double intervals_per_integer = 1;

        /// <summary>Polynomial interpolation interpolation points per interval</summary>
        public int n_interpolation_points = 3;
    }
}
