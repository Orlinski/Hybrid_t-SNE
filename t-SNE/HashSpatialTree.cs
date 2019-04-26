using System;
using System.Threading.Tasks;

namespace Hybrid_tSNE
{
    public class HashSpatialTree
    {
        public static int mode = 0;
        private HSTNode root;
        private double size;
        private readonly int N, D;
        private double[] points;
        private int[] ids;
        private ulong[] hashes;
        private ulong[] morton;
        private readonly int bit4dim;
        private readonly ulong maxv; //2^bit4dim


        public HashSpatialTree(int N, int D, double[] points) //j + i * D; i - points; j - dims
        {
            PrepareMortonLookup(D);
            this.points = points;
            this.N = N;
            this.D = D;

            bit4dim = 64 / D;
            maxv = 1UL << bit4dim; //2^bit4dim

            ids = new int[N];
            hashes = new ulong[N];
        }

        public void RebuildTree(bool presort)
        {
            size = 2 * AbsoluteMax();

            Parallel.For(0, N, i =>
            {
                ids[i] = i;
                ulong hash = 0;
                for (int j = 0; j < D; j++)
                {
                    ulong bin = (ulong)(maxv * (points[j + i * D] / size + 0.5)); //normalization should be <0, 1)
                    if (bin == maxv) bin--; //fix normalization
                    for (int k = j; bin != 0; k += 8 * D)
                    {
                        hash += morton[bin & 0xFF] << k;
                        bin >>= 8;
                    }
                }
                hashes[i] = hash;
            });

            if (presort)
            {
                Array.Sort(hashes, ids);
                root = BuildTreePresorted(bit4dim * D - 1, 0, hashes.Length);
            }
            else
                root = BuildTreeParallel(bit4dim * D - 1, 0, hashes.Length);

            root.Fill(size / 2.0);
        }

        private double AbsoluteMax()
        {
            double max = 0;
            foreach (double value in points)
            {
                if (max < value)
                    max = value;
                else if (max < -value)
                    max = -value;
            }
            return max;
        }

        private void PrepareMortonLookup(int D)
        {
            morton = new ulong[256];
            for (int i = 1; i < 256; i++)
            {
                ulong val = (ulong)i;
                ulong hash = 0;
                for (int k = 0; val != 0; k += D)
                {
                    hash += (val & 1UL) << k;
                    val >>= 1;
                }
                morton[i] = hash;
            }
        }

        private HSTNode BuildTreeParallel(int bit, int left, int right, int level = 0)
        {
            if (bit < 0 || right - left <= 1) return null;

            int[] dids = new int[(1 << D) + 1];
            dids[0] = left;
            dids[1 << D] = right;
            SortAndDivide(dids, bit, 0, 1 << D);

            HSTNode[] children = new HSTNode[1 << D];
            if (level < 3 && right - left > 2048)
                Parallel.For(0, 1 << D, i => children[i] = BuildTreeParallel(bit - D, dids[i], dids[i + 1], level + 1));
            else
                for (int i = (1 << D) - 1; i >= 0; i--) children[i] = BuildTreeParallel(bit - D, dids[i], dids[i + 1], level + 1);

            return new HSTNode(this, dids, children);
        }

        private void SortAndDivide(int[] dids, int bit, int start, int end)
        {
            int mid = (start + end) / 2;
            dids[mid] = Pivot(bit, dids[start], dids[end]);
            if (mid - start > 1)
            {
                SortAndDivide(dids, bit - 1, start, mid);
                SortAndDivide(dids, bit - 1, mid, end);
            }
        }

        private int Pivot(int bit, int left, int right)
        {
            ulong mask = 1UL << bit;
            int l = left;
            int r = right - 1;
            while (l < r)
            {
                while ((hashes[l] & mask) == 0 && l < r) l++;
                while ((hashes[r] & mask) != 0 && l < r) r--;

                if (l < r)
                {
                    int id = ids[l];
                    ids[l] = ids[r];
                    ids[r] = id;
                    ulong hash = hashes[l];
                    hashes[l] = hashes[r];
                    hashes[r] = hash;
                }
                else
                    break;
            }

            return l;
        }

        private HSTNode BuildTreePresorted(int bit, int left, int right)
        {
            if (bit < 0 || right - left <= 1) return null;

            int[] dids = new int[(1 << D) + 1];
            dids[0] = left;
            dids[1 << D] = right;
            dids = FindDivision(dids, bit, 0, 1 << D);

            HSTNode[] children = new HSTNode[1 << D];
            for (int i = (1 << D) - 1; i >= 0; i--) children[i] = BuildTreePresorted(bit - D, dids[i], dids[i + 1]);

            return new HSTNode(this, dids, children);
        }

        private int[] FindDivision(int[] dids, int bit, int start, int end)
        {
            ulong div = 1UL << bit;
            int mid = (start + end) / 2;
            dids[mid] = HashBinarySearch(div, dids[start], dids[end]);
            if (mid - start > 1)
            {
                FindDivision(dids, bit - 1, start, mid);
                FindDivision(dids, bit - 1, mid, end);
            }
            return dids;
        }

        private int HashBinarySearch(ulong bitmask, int left, int right)
        {
            if (right == left) return left;
            if (right - left == 1)
                if ((hashes[left] & bitmask) == 0) return right;
                else return left;

            int s = (left + right) / 2;
            if ((hashes[s] & bitmask) == 0) return HashBinarySearch(bitmask, s, right);
            return HashBinarySearch(bitmask, left, s);
        }

        public double EstimateRepulsion(double[] GradR, double Theta2)
        {
            double Z = 0;
            object Zlock = new object();

            Parallel.For(0, N,
            () => 0.0d,
            (i, loopState, partialZ) =>
            {
                return root.EstimateRepulsion(i, GradR, Theta2) + partialZ;
            },
            (localPartialZ) =>
            {
                lock (Zlock)
                {
                    Z += localPartialZ;
                }
            });

            return Z;
        }

        class HSTNode
        {
            private readonly HashSpatialTree tree;
            private readonly int D;
            internal readonly int[] DIds;
            internal readonly HSTNode[] children;
            private readonly double[] COM;
            private double HSize2; //squared cell half size
            internal int Count;

            public HSTNode(HashSpatialTree tree)
            {
                this.tree = tree;
                this.D = tree.D;
                this.DIds = new int[1 << D + 1];
                this.children = new HSTNode[1 << D];
                COM = new double[D];
            }

            public HSTNode(HashSpatialTree tree, int[] DIds, HSTNode[] children)
            {
                this.tree = tree;
                this.D = tree.D;
                this.DIds = DIds;
                Count = DIds[1 << D] - DIds[0];
                this.children = children;
                COM = new double[D];
            }


            public void Fill(double hsize)
            {
                HSize2 = hsize * hsize;
                Array.Clear(COM, 0, D);

                for (int k = (1 << D) - 1; k >= 0; k--)
                    if (children[k] == null)
                    {
                        for (int i = DIds[k]; i < DIds[k + 1]; i++)
                        {
                            int id = tree.ids[i];
                            for (int j = 0; j < D; j++) COM[j] += tree.points[j + id * D];
                        }
                    }
                    else
                    {
                        children[k].Fill(hsize / 2);
                        for (int j = 0; j < D; j++) COM[j] += children[k].Count * children[k].COM[j];
                    }

                for (int j = 0; j < D; j++) COM[j] /= Count;
            }

            public double EstimateRepulsion(int pid, double[] grad, double theta2)
            {
                int id = pid * D;
                double diff;
                double tdist = 0;
                for (int k = 0; k < D; k++)
                {
                    diff = tree.points[k + id] - COM[k];
                    tdist += diff * diff; //real distance
                }
                if (HSize2 < theta2 * tdist)
                {
                    tdist = 1 / (1 + tdist);
                    double weight = Count * tdist * tdist;
                    for (int k = 0; k < D; k++) grad[k + id] += weight * (tree.points[k + id] - COM[k]);
                    return tdist * Count;
                }

                tdist = 0;
                for (int i = (1 << D) - 1; i >= 0; i--)
                    if (children[i] != null) tdist += children[i].EstimateRepulsion(pid, grad, theta2);
                    else
                        for (int j = DIds[i]; j < DIds[i + 1]; j++) tdist += EstimateRepulsion(pid, tree.ids[j], grad);

                return tdist;
            }

            private double EstimateRepulsion(int pid, int pid2, double[] grad)
            {
                if (pid2 == pid) return 0;
                pid *= D;
                pid2 *= D;

                double tdist = 0;
                double diff;
                for (int k = 0; k < D; k++)
                {
                    diff = tree.points[k + pid] - tree.points[k + pid2];
                    tdist += diff * diff; //real distance
                }

                tdist = 1 / (1 + tdist);

                double weight = tdist * tdist;
                for (int k = 0; k < D; k++)
                    grad[k + pid] += weight * (tree.points[k + pid] - tree.points[k + pid2]);

                return tdist;
            }
        }
    }
}