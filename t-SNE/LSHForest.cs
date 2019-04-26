using Hybrid_tSNE.Ordering;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Hybrid_tSNE
{
    internal static class LSHForest
    {
        public static void SymmetricANN(float[][] data, int k, LSHFConfiguration LSHFConfig, out List<int>[] ids, out List<double>[] dists, bool verbose = false)
        {
            ids = Candidates(data, k, LSHFConfig, verbose);

            if (verbose) Console.WriteLine("Choosing neighbours from candidates");
            dists = BestCandidates(ids, data, k);

            if (verbose) Console.WriteLine("Symmetrizing neighbours");
            Symmetrize(ids, dists, k);
        }

        public static void ANN(float[][] data, int k, LSHFConfiguration LSHFConfig, out List<int>[] ids, out List<double>[] dists, bool verbose = false)
        {
            ids = Candidates(data, k, LSHFConfig, verbose);

            if (verbose) Console.WriteLine("Choosing neighbours from candidates");
            dists = BestCandidates(ids, data, k);
        }

        public static List<int>[] Candidates(float[][] data, int k, LSHFConfiguration LSHFConfig, bool verbose = false)
        {
            Random SeedGen = LSHFConfig.LSHSeed == -1 ? new Random() : new Random(LSHFConfig.LSHSeed);
            int N = data.Length;
            int trees = LSHFConfig.LSHForestTrees;

            List<int>[] candidates = new List<int>[N];
            if (N < 2 * k)
            {
                for (int i = 0; i < N; i++)
                {
                    candidates[i] = Enumerable.Range(0, N).ToList();
                    candidates[i].RemoveAt(i);
                }
                return candidates;
            }

            for (int i = 0; i < N; i++) candidates[i] = new List<int>(2 * k);
            int[] seeds = new int[trees];
            for (int i = 0; i < trees; i++)
                seeds[i] = SeedGen.Next();

            if (verbose) Console.WriteLine("LSH Forest building {0} trees", trees);

            Parallel.For(0, trees, i =>
            {
                int[] indicies = LSHTree.InMemoryGet(data, LSHFConfig.LSHTreeC * k / trees, LSHFConfig.LSHHashDims, seeds[i], out int[] cand);

                for (int j = 0; j < N; j++)
                    lock (candidates[j])
                        for (int ii = indicies[2 * j]; ii < indicies[2 * j + 1]; ii++)
                            candidates[j].Add(cand[ii]);
            });

            Parallel.For(0, N, i =>
            {
                candidates[i].Sort();
                int a = 0;
                int b = 1;
                while (b < candidates[i].Count)
                    if (candidates[i][b] == candidates[i][a]) b++;
                    else candidates[i][++a] = candidates[i][b++];
                a++;
                candidates[i].RemoveRange(a, b - a);
                candidates[i].Remove(i);
            });

            int fill = RandomFill(candidates, k, SeedGen.Next());
            if (fill != 0) Console.WriteLine("WARNING! Added {0} random candidates total in a LSH Forest.", fill);
            return candidates;
        }

        private static int RandomFill(List<int>[] candidates, int k, int seed)
        {
            Random R = new Random(seed);

            int fill = 0;
            int N = candidates.Length;
            for (int i = 0; i < N; i++)
            {
                if (candidates[i].Count < k)
                {
                    candidates[i].Sort();
                    HashSet<int> add = new HashSet<int>();
                    int count = candidates[i].Count;

                    while (add.Count < k - count)
                    {
                        int c = R.Next(0, N);
                        if (c != i && !BinarySearch(c, candidates[i], count) && !add.Contains(c)) add.Add(c);
                    }

                    candidates[i].AddRange(add);
                    fill += add.Count;
                }
            }
            return fill;
        }

        public static List<double>[] BestCandidates(List<int>[] candidates, float[][] data, int k)
        {
            int N = candidates.Length;
            List<double>[] dists = new List<double>[N];

            Parallel.For(0, N, i =>
            {
                dists[i] = candidates[i].Select(id => SqrEuclid(data[id], data[i])).ToList();
                Selection.Quickselect(candidates[i], dists[i], k);
                candidates[i].RemoveRange(k, candidates[i].Count - k);
                dists[i].RemoveRange(k, dists[i].Count - k);
            });

            return dists;
        }

        private static void Symmetrize(List<int>[] ids, List<double>[] dists, int k)
        {
            int N = ids.Length;

            Parallel.For(0, N, i => { Quicksort<int, double>.Sort(dists[i], ids[i]); });

            Parallel.For(0, N, i =>
            {
                for (int j = 0; j < k; j++)
                {
                    int id = ids[i][j];
                    if (!BinarySearch(i, ids[id], k))
                        lock (ids[id])
                        {
                            ids[id].Add(i);
                            dists[id].Add(dists[i][j]);
                        }
                }
            });
        }

        private static bool BinarySearch(int id, IList<int> ids, int k)
        {
            int left = 0;
            int right = k;
            while (left <= right && left < k)
            {
                int mid = (left + right) / 2;
                if (id == ids[mid]) return true;
                else if (id < ids[mid]) right = mid - 1;
                else left = mid + 1;
            }
            return false;
        }

        public static double SqrEuclid(float[] a, float[] b)
        {
            double res = 0;
            double tmp;
            for (int i = a.Length - 1; i >= 0; --i)
            {
                tmp = a[i] - b[i];
                res += tmp * tmp;
            }
            return res;
        }
    }

    internal class LSHTree
    {
        private readonly Random R;
        private readonly int Hdims;
        private readonly float[][] data;
        private readonly int D;
        private readonly int count;
        private int[] ids;
        private readonly double[] hashes;
        private int[] indicies;
        private const float chunk = 0.1f;
        private const float chunk50 = 1 - (chunk + 0.5f) / 2;

        public static int[] InMemoryGet(float[][] data, int count, int Hdims, int seed, out int[] ids)
        {
            LSHTree instance = new LSHTree(data, count, Hdims, seed);
            instance.InMemoryGet(0, data.Length);
            ids = instance.ids;
            return instance.indicies;
        }

        private LSHTree(float[][] data, int count, int Hdims, int seed)
        {
            R = new Random(seed);
            this.Hdims = Hdims;
            this.data = data;
            D = data[0].Length;
            this.count = count;
            int N = data.Length;
            ids = Enumerable.Range(0, N).ToArray();
            hashes = new double[N];
            indicies = new int[2 * N];
        }

        private void InMemoryGet(int left, int right) //<left, right)
        {
            IHashVector hv = new HashVector(D, Hdims, R);
            for (int i = left; i < right; i++) hashes[i] = hv.Hash(data[ids[i]]);

            int div = Selection.RoughQuickselect(ids, hashes, left, right, chunk);

            if ((div - left) * chunk50 > count) InMemoryGet(left, div);
            else
            {
                int lim = (div - left < count) ? right : div;
                for (int i = left; i < div; i++)
                {
                    indicies[2 * ids[i]] = left;
                    indicies[2 * ids[i] + 1] = lim;
                }
            }

            if ((right - div) * chunk50 > count) InMemoryGet(div, right);
            else
            {
                int lim = (right - div < count) ? left : div;
                for (int i = div; i < right; i++)
                {
                    indicies[2 * ids[i]] = lim;
                    indicies[2 * ids[i] + 1] = right;
                }
            }
        }
    }

    internal interface IHashVector
    {
        double Hash(float[] data);
    }

    internal struct HashVector : IHashVector
    {
        public static long hv_no = 0;
        private readonly KeyValuePair<int, double>[] vector;

        public HashVector(int D, int Dims, Random R)
        {
            hv_no += 1;
            vector = new KeyValuePair<int, double>[Dims];
            for (int i = 0; i < Dims; i++) vector[i] = new KeyValuePair<int, double>(R.Next(D), 2 * R.NextDouble() - 1);
        }

        public double Hash(float[] data)
        {
            double val = 0;
            foreach (KeyValuePair<int, double> v in vector) val += data[v.Key] * v.Value;
            return val;
        }
    }
}