using System;
using System.Collections.Generic;

namespace Hybrid_tSNE.Ordering
{
    /// <summary>
    /// Sort implementation for two ILists of keys and values based of Array.Sort
    /// </summary>
    /// <typeparam name="V"></typeparam>
    /// <typeparam name="T"></typeparam>
    public static class Quicksort<V, T> where V : IComparable
    {
        /// <summary>
        /// Sorts a pair of one-dimensional IList objects (one contains the keys and the other contains the corresponding items) based on the values in second IList using IComparable implementation of each value.
        /// </summary>
        /// <param name="elements"></param>
        /// <param name="values"></param>
        public static void Sort(IList<T> elements, IList<V> values)
        {
            Sort(elements, values, 0, elements.Count - 1, 2 * (int)Math.Log(elements.Count));
        }

        /// <summary>
        /// Sorts a range of elements in a pair of one-dimensional IList objects (one contains the keys and the other contains the corresponding items) based on the values in second IList using IComparable implementation of each value.
        /// </summary>
        /// <param name="elements"></param>
        /// <param name="values"></param>
        /// <param name="index"></param>
        /// <param name="length"></param>
        public static void Sort(IList<T> elements, IList<V> values, int index, int length)
        {
            if (length < 2) return;
            Sort(elements, values, index, length + index - 1, 2 * (int)Math.Log(elements.Count));
        }

        private static void SwapIfGreater(IList<T> elements, IList<V> values, int a, int b)
        {
            if (a == b || values[a].CompareTo(values[b]) < 0) return;
            V val = values[a];
            values[a] = values[b];
            values[b] = val;
            T id = elements[a];
            elements[a] = elements[b];
            elements[b] = id;
        }

        private static void Swap(IList<T> elements, IList<V> values, int a, int b)
        {
            if (a == b) return;
            V val = values[a];
            values[a] = values[b];
            values[b] = val;
            T id = elements[a];
            elements[a] = elements[b];
            elements[b] = id;
        }

        private static void Sort(IList<T> elements, IList<V> values, int left, int right, int depthLimit)
        {
            for (int pivot; left < right; right = pivot - 1)
            {
                int span = right - left + 1;
                if (span <= 16)
                {
                    if (span == 1) break;
                    if (span == 2)
                    {
                        SwapIfGreater(elements, values, left, right);
                        break;
                    }
                    if (span == 3)
                    {
                        SwapIfGreater(elements, values, left, right - 1);
                        SwapIfGreater(elements, values, left, right);
                        SwapIfGreater(elements, values, right - 1, right);
                        break;
                    }
                    InsertionSort(elements, values, left, right);
                    break;
                }
                if (depthLimit == 0)
                {
                    Heapsort(elements, values, left, right);
                    break;
                }
                --depthLimit;
                pivot = PickPivotAndPartition(elements, values, left, right);
                Sort(elements, values, pivot + 1, right, depthLimit);
            }
        }

        private static int PickPivotAndPartition(IList<T> elements, IList<V> values, int left, int right)
        {
            int pivot = left + (right - left) / 2;
            SwapIfGreater(elements, values, left, pivot);
            SwapIfGreater(elements, values, left, right);
            SwapIfGreater(elements, values, pivot, right);
            V val = values[pivot];
            Swap(elements, values, pivot, right - 1);
            int i = left;
            int j = right - 1;
            while (i < j)
            {
#pragma warning disable CS0642 // Possible mistaken empty statement
                do
                    ;
                while (values[++i].CompareTo(val) < 0);
                do
                    ;
                while (val.CompareTo(values[--j]) < 0);
#pragma warning restore CS0642 // Possible mistaken empty statement
                if (i < j)
                    Swap(elements, values, i, j);
                else
                    break;
            }
            Swap(elements, values, i, right - 1);
            return i;
        }

        private static void Heapsort(IList<T> elements, IList<V> values, int left, int right)
        {
            int n = right - left + 1;
            for (int i = n / 2; i >= 1; --i)
                DownHeap(elements, values, i, n, left);
            for (int i = n; i > 1; --i)
            {
                Swap(elements, values, left, left + i - 1);
                DownHeap(elements, values, 1, i - 1, left);
            }
        }

        private static void DownHeap(IList<T> elements, IList<V> values, int i, int n, int left)
        {
            V val = values[left + i - 1];
            T id = elements[left + i - 1];
            for (int j; i <= n / 2; i = j)
            {
                j = 2 * i;
                if (j < n && values[left + j - 1].CompareTo(values[left + j]) < 0)
                    ++j;
                if (val.CompareTo(values[left + j - 1]) < 0)
                {
                    values[left + i - 1] = values[left + j - 1];
                    if (elements != null)
                        elements[left + i - 1] = elements[left + j - 1];
                }
                else
                    break;
            }
            values[left + i - 1] = val;
            if (elements == null)
                return;
            elements[left + i - 1] = id;
        }

        private static void InsertionSort(IList<T> elements, IList<V> values, int left, int right)
        {
            for (int i = left; i < right; ++i)
            {
                int j = i;
                V val = values[i + 1];
                T id = elements[i + 1];
                for (; j >= left && val.CompareTo(values[j]) < 0; --j)
                {
                    values[j + 1] = values[j];
                    if (elements != null)
                        elements[j + 1] = elements[j];
                }
                values[j + 1] = val;
                if (elements != null)
                    elements[j + 1] = id;
            }
        }
    }
}
