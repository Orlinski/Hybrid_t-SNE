using System.Collections.Generic;

namespace Hybrid_tSNE.Ordering
{
    public struct Selection
    {
        private readonly IList<int> ids;
        private readonly IList<double> vals;

        private Selection(IList<int> ids, IList<double> vals)
        {
            this.ids = ids;
            this.vals = vals;
        }

        /// <summary>
        /// Does quickselect on a pair of int and double ILists. k-th smallest element ends up on k-th place and all smaller are to its left.
        /// </summary>
        /// <param name="ids"></param>
        /// <param name="vals"></param>
        /// <param name="k"></param>
        public static void Quickselect(IList<int> ids, IList<double> vals, int k)
        {
            new Selection(ids, vals).Select(0, ids.Count - 1, k - 1);
        }

        /// <summary>
        ///  Does quickselect on a pair of int and double ILists, until smaller partition is at least chunk in size.
        /// </summary>
        /// <param name="ids"></param>
        /// <param name="vals"></param>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <param name="chunk"></param>
        /// <returns>Index of pivot.</returns>
        public static int RoughQuickselect(IList<int> ids, IList<double> vals, int left, int right, float chunk)
        {
            int k = (int)((right - left) * chunk);
            Selection s = new Selection(ids, vals);
            return s.RoughSelect(left, right - 1, left + k, right - k);
        }

        private int Partition(int left, int right, int pivoti)
        {
            //swap pivoti and right in vals
            double pivotv = vals[pivoti];
            vals[pivoti] = vals[right];
            int pivotid = ids[pivoti];
            ids[pivoti] = ids[right];
            //vals[right] = pivotv; //will be replaced anyway
            //ids[right] = pivotid;

            //move all smaller than pivot to front
            int currswap = left;
            for (int i = left; i < right; i++)
                if (pivotv > vals[i])
                {
                    double temp = vals[currswap];
                    vals[currswap] = vals[i];
                    vals[i] = temp;
                    int tempid = ids[currswap];
                    ids[currswap] = ids[i];
                    ids[i] = tempid;
                    currswap++;
                }

            //move pivot into position
            vals[right] = vals[currswap];
            vals[currswap] = pivotv;
            ids[right] = ids[currswap];
            ids[currswap] = pivotid;

            return currswap;
        }

        private void SwapIfGreater(int a, int b)
        {
            if (vals[b] > vals[a])
            {
                double temp = vals[a];
                vals[a] = vals[b];
                vals[b] = temp;
                int tempid = ids[a];
                ids[a] = ids[b];
                ids[b] = tempid;
            }
        }

        private void Select(int left, int right, int n)
        {
            int pivoti;
            while (left < right)
            {
                pivoti = left + (right - left) / 2;
                SwapIfGreater(left, pivoti);
                SwapIfGreater(left, right);
                SwapIfGreater(pivoti, right);

                pivoti = Partition(left, right, pivoti);

                if (n == pivoti) return;
                else if (n < pivoti) right = pivoti - 1;
                else left = pivoti + 1;
            }
        }

        private int RoughSelect(int left, int right, int min, int max)
        {
            int pivoti = left;
            while (left < right)
            {
                pivoti = left + (right - left) / 2;
                SwapIfGreater(left, pivoti);
                SwapIfGreater(left, right);
                SwapIfGreater(pivoti, right);

                pivoti = Partition(left, right, pivoti);

                if (pivoti < min) left = pivoti + 1;
                else if (pivoti > max) right = pivoti - 1;
                else return pivoti;
            }
            return pivoti;
        }
    }
}