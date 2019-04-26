using Hybrid_tSNE;
using Mono.Options;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;


namespace tSNE_Runner
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            CultureInfo.DefaultThreadCurrentCulture = new CultureInfo("en-US");
            
            string config_file = "";
            bool output_csv = false;
            bool verbose = false;

            bool show_help = false;
            var p = new OptionSet()
            {
                { "h|help", "show this message and exit", v => show_help = v != null },
                { "v|verbose", "show information about progress", v => verbose = v != null },
                { "c|config=", "configuration file to use", v => config_file = v },
                { "f|format=", "output format - binary(default)/csv", v => output_csv = v == "csv" }
            };

            List<string> extra;
            try
            {
                extra = p.Parse(args);
            }
            catch (OptionException e)
            {
                Console.Write("Hybrid_t-SNE: ");
                Console.WriteLine(e.Message);
                Console.WriteLine("Try `Hybrid_t-SNE --help' for more information.");
                return;
            }

            if (show_help || extra.Count != 3)
            {
                ShowHelp(p);
                return;
            }

            Console.WriteLine("Output dimensions:\t{0}", extra[0]);
            Console.WriteLine("Input file:\t{0}", extra[1]);
            Console.WriteLine("Output file:\t{0}", extra[2]);
            Console.WriteLine(output_csv ? "Output as CSV" : "Binary output");
            if (verbose)
                Console.WriteLine("Verbose progress");
            if (config_file != "")
                Console.WriteLine("Using Hybrid t-SNE parameters from file:\t{0}", config_file);

            Run(extra[1], extra[2], output_csv, config_file, Convert.ToInt32(extra[0]), verbose);
        }

        private static void Run(string input_file, string output_file, bool output_csv, string config_file, int dimensions, bool verbose)
        {
            float[][] Data = ReadBinary(input_file);

            Stopwatch sw = Stopwatch.StartNew();
            tSNE tsne = new tSNE(Data);
            sw.Stop();
            if (config_file != "")
                tsne = SetConfig(tsne, config_file);
            sw.Start();
            double[][] Y = tsne.Reduce(dimensions, verbose);
            sw.Stop();
            Console.WriteLine("t-SNE reduction time:\t{0}s", sw.ElapsedMilliseconds / 1000.0);

            if (output_csv)
                SaveCSV(Y, output_file);
            else
                SaveBinary(Y, output_file);
        }

        private static void ShowHelp(OptionSet p)
        {
            Console.WriteLine("Usage: Hybrid_t-SNE [OPTIONS]+ output_dimensions input_file output_file");
            Console.WriteLine("Runs Hybrid t-SNE algorithm.");
            Console.WriteLine();
            Console.WriteLine("Options:");
            p.WriteOptionDescriptions(Console.Out);
        }

        private static float[][] ReadBinary(string filename)
        {
            float[][] data;

            using (BinaryReader b = new BinaryReader(File.Open(filename, FileMode.Open)))
            {
                long n = b.ReadInt32();
                long d = b.ReadInt32();

                data = new float[n][];
                for (int i = 0; i < n; i++)
                {
                    data[i] = new float[d];
                    for (int j = 0; j < d; j++)
                        data[i][j] = b.ReadSingle();
                }
            }

            return data;
        }

        private static void SaveBinary(double[][] Data, string filename)
        {
            int N = Data.Length;
            int D = Data[0].Length;

            using (BinaryWriter b = new BinaryWriter(File.Open(filename, FileMode.Create)))
            {
                b.Write(N);
                b.Write(D);

                for (int i = 0; i < N; i++)
                    for (int j = 0; j < D; j++)
                        b.Write(Data[i][j]);
            }
        }

        private static void SaveCSV(double[][] Data, string filename)
        {
            using (StreamWriter writer = new StreamWriter(filename))
                for (int i = 0; i < Data.Length; i++)
                    writer.WriteLine(string.Join(",", Data[i]));
        }

        public static tSNE SetConfig(tSNE tsne, string configFile)
        {
            string json;
            try
            {
                json = File.ReadAllText(configFile);
            }
            catch (FileNotFoundException)
            {
                Console.WriteLine("Configuration file {0} not found! Using default parameters.");
                return tsne;
            }

            Dictionary<string, Dictionary<string, object>> config = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, object>>>(json);

            foreach (KeyValuePair<string, Dictionary<string, object>> conf in config)
                UpdateConfig(tsne, conf.Key, conf.Value);
            
            Console.WriteLine("\n\n");

            return tsne;
        }

        private static void UpdateConfig(tSNE tsne, string conf, Dictionary<string, object> param)
        {
            Console.WriteLine("\n{0}Configuration:", conf);
            switch (conf)
            {
                case "Initialization":
                    InitializationConfiguration ic = tsne.InitializationConfig;
                    ic.SmartInit = Convert.ToBoolean(UpdateField(param, "SmartInit", ic.SmartInit));
                    ic.InitialSolutionSeed = Convert.ToInt32(UpdateField(param, "InitialSolutionSeed", ic.InitialSolutionSeed));
                    if (ic.InitialSolutionSeed == -1)
                        Console.WriteLine("\tInitialSolutionSeed set to -1 - using random seed.");
                    break;
                case "Affinities":
                    AffinitiesConfiguration ac = tsne.AffinitiesConfig;
                    ac.Perplexity = Convert.ToDouble(UpdateField(param, "Perplexity", ac.Perplexity));
                    ac.EntropyTol = Convert.ToDouble(UpdateField(param, "EntropyTol", ac.EntropyTol));
                    ac.EntropyIter = Convert.ToInt32(UpdateField(param, "EntropyIter", ac.EntropyIter));
                    break;
                case "LSHF":
                    LSHFConfiguration lc = tsne.LSHFConfig;
                    lc.LSHForestTrees = Convert.ToInt32(UpdateField(param, "LSHForestTrees", lc.LSHForestTrees));
                    lc.LSHTreeC = Convert.ToInt32(UpdateField(param, "LSHTreeC", lc.LSHTreeC));
                    lc.LSHHashDims = Convert.ToInt32(UpdateField(param, "LSHHashDims", lc.LSHHashDims));
                    lc.LSHSeed = Convert.ToInt32(UpdateField(param, "LSHSeed", lc.LSHSeed));
                    if (lc.LSHSeed == -1)
                        Console.WriteLine("\tLSHSeed set to -1 - using random seed.");
                    break;
                case "Gradient":
                    GradientConfiguration gc = tsne.GradientConfig;
                    gc.Iterations = Convert.ToInt32(UpdateField(param, "Iterations", gc.Iterations));
                    gc.GradMinGain = Convert.ToDouble(UpdateField(param, "GradMinGain", gc.GradMinGain));
                    gc.RepulsionMethod = UpdateRepulsionMethod(param, gc.RepulsionMethod);
                    gc.Exaggeration = UpdateFunctionField(param, "Exaggeration", gc.Exaggeration);
                    gc.Momentum = UpdateFunctionField(param, "Momentum", gc.Momentum);
                    gc.LearningRate = UpdateFunctionField(param, "LearningRate", gc.LearningRate);
                    break;
                case "BarnesHut":
                    BarnesHutConfiguration bc = tsne.BarnesHutConfig;
                    bc.BarnesHutCondition = Convert.ToDouble(UpdateField(param, "BarnesHutCondition", bc.BarnesHutCondition));
                    bc.Presort = Convert.ToBoolean(UpdateField(param, "Presort", bc.Presort));
                    break;
                case "PI":
                    PIConfiguration pc = tsne.PIConfig;
                    pc.min_num_intervals = Convert.ToInt32(UpdateField(param, "min_num_intervals", pc.min_num_intervals));
                    pc.intervals_per_integer = Convert.ToDouble(UpdateField(param, "intervals_per_integer", pc.intervals_per_integer));
                    pc.n_interpolation_points = Convert.ToInt32(UpdateField(param, "n_interpolation_points", pc.n_interpolation_points));
                    break;
                default:
                    Console.WriteLine("\tConfiguration type {0} unknown.", conf);
                    break;
            }
            if (param.Count > 0)
                Console.WriteLine("\tUnknown {0}Configuration parameters: {1}!", conf, string.Join(", ", param.Keys));
        }

        private static object UpdateField(Dictionary<string, object> param, string parameter, object value)
        {
            if (param.TryGetValue(parameter, out object val))
            {
                param.Remove(parameter);
                Console.WriteLine("\tSet {0} to:\t{1}", parameter, Convert.ChangeType(val, value.GetType()));
                return val;
            }
            return value;
        }

        private static RepulsionMethods UpdateRepulsionMethod(Dictionary<string, object> param, RepulsionMethods value)
        {
            string parameter = "RepulsionMethod";
            if (param.TryGetValue(parameter, out object val))
            {
                param.Remove(parameter);
                switch (Convert.ToString(val))
                {
                    case "auto":
                        value = RepulsionMethods.auto;
                        break;
                    case "barnes_hut":
                        value = RepulsionMethods.barnes_hut;
                        break;
                    case "fft":
                        value = RepulsionMethods.fft;
                        break;
                    default:
                        Console.WriteLine("\tUnknown value {} for RepulsionMethod!");
                        break;
                }
                Console.WriteLine("\tSet {0} to:\t{1}", parameter, value);
            }
            return value;
        }

        private static Func<int, int, double> UpdateFunctionField(Dictionary<string, object> param, string parameter, Func<int, int, double> value)
        {
            if (param.TryGetValue(parameter, out object val))
            {
                param.Remove(parameter);
                string definition = Convert.ToString(val);
                value = Function.MakeFunction(Convert.ToString(val));
                Console.WriteLine("Set {0} to:\t{1}", parameter, definition);
            }
            return value;
        }
    }
}
