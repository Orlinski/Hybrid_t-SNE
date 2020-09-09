# Hybrid t-SNE
Hybrid t-SNE is currently the fastest implementation of t-SNE dimensionality reduction algorithm. Hybrid t-SNE has O(m log m) complexity, thanks to utilization of Localily Sensitive Hashing (LSH) Forest of balaced trees for nearest neighbors search. Its speed is supported by an automatic selection of repulsive gradient computation method, between Barnes-Hut approximation and Polynomial Interpolation approximation.

We provide three versions of Hybrid t-SNE:
* .NET Framework 4.5 version - Adapted directly from our article, with minor changes, for ease of use. Fastest, when used natively on Windows.
* .NET Core 2.2 version - Made for multiplatform compatibility. Ported from .NET Framework version without changes.
* "Readable" version (unmaintained) - Code easier to understand, with some optimalizations removed, like dimensionality dedicated methods and flattening of multidimensional arrays. More comprehensible, but slower - not recommended for use.


## Dependencies:
.NET Framework 4.5 or .NET Core 2.2

### Requirements (via NuGet)
Hybrid t-SNE:
* MathNet.Numerics
* MathNet.Numerics.MKL

Runner (native wrapper):
* Mono.Options (parsing command line arguments)
* Newtonsoft.Json (parsing config file)
* CoreCLR-NCalc (parsing mathematical expressions to functions for some configurable parameters)

*Note:* Intel Math Kernel Library (MKL) is required as long as Fourier.ForwardMultiDim is not implemented in managed MathNet.Numerics, but even when it's available, MKL is highly recommended.

## Building
Visual Studio 2017 is recommended for building. Just open the solution and build!

*Note:* You may have to remove one of MKL dependencies in t-SNE project, depending on target platform.

## Runner (native wrapper) usage
Prerequisites: .NET Framework or .NET Core, Intel MKL.

Usage: `Hybrid_t-SNE [OPTIONS]+ output_dimensions input_file output_file`

Note: for .NET Core preface above with dotnet

Options:\
  `-h, --help`                 help and exits\
  `-v, --verbose`              show information about progress\
  `-c VALUE, --config=VALUE`         configuration file to use\
  `-f VALUE, --format=VALUE`         output format - binary(default)/csv
  
`output_dimensions` - number of dimensions to reduce to - 1, 2 or 3. *Note:* higher values will work, but are not recommended (this is not what t-SNE is for)\
`input_file` - path of file to read dataset from\
`output_file` - path of file to save reduced dataset to

Example: `Hybrid_t-SNE -v -c config.json -f csv 2 dataset.bin reduced.csv`\
This will run reduction of `dataset.bin` to 2 dimensions, using configuration from `config.json`, printing progress and save result to csv file `reduced.csv`.

### Input/output format
Binary files are in simplest format possible:
1. First there are two Int32 numbers - N (number of instances) and D (number of attributes).
2. Next there are N * D Float(input)/Double(output) numbers with values instance by instance. Therefore first D numbers are attribute values for first instance, next D for second instance, and so on.

CSV file format:
Each line is an instance, attribute values are separated by commas, decimal separator is a dot.

## Configuration file
There are multiple configurable parameters for Hybrid t-SNE. They are divided into groups to make configuration more manageable.\
When using Runner, you can set all parameters by providing JSON configuration file. Example file with default values is provided as config.json.\
When configuration file is not provided defaults are used. When any of the values is not provided, the default is used as well, therefore you may only specify parameters you want to change.

Most parameters are int, float or bool, as easy to see from example config. Some Gradient parameters are however special:
1. *RepulsionMethod* can be set to: "*auto*", "*barnes_hut*" or "*fft*". This changes repulsion gradient used and therefore for:
	- "*auto*" - Hybrid t-SNE is run - better method is chosen automatically
	- "*barnes_hut*" - LSHF BH t-SNE is run
	- "*fft*" - PI t-SNE is run
2. *Exaggeration*, *Momentum* and *LearningRate* are provided as functions of 2 variables - *Iteration* (current gradient step) and *Iterations* (set number of gradient steps). See defaults provided in `config.json` and [NCalc documentation](https://github.com/ncalc/ncalc/wiki/Functions), to make your own function.

## Using Class Library directly
See Runner for example use, in short:
1. Make t-SNE instance, providing data - `float[][]` (first index is instance, second is attribute): `tSNE tsne = new tSNE(Data);`
2. Set parameters - see Config.cs for Configuration classes and defaults: e.g. `tsne.GradientConfig.Iterations = 1000;`
3. Run reduction - `double[][] Y = tsne.Reduce(output_dimensionality);`

## Authors
Marek Orliński and Norbert Jankowski, [Department of Informatics](https://www.fizyka.umk.pl/en/doi/), [Nicolaus Copernicus University in Toruń](https://www.umk.pl/en/)

## Citation
Please cite our work when using our algorithm:

```
Marek Orliński and Norbert Jankowski. “Fast t-SNE algorithm with forest of balanced LSH trees and hybrid computation of repulsive forces”. In: Knowledge-Based Systems 208 (2020), pp. 1–16
```

```
@ARTICLE{OrlinskiJankowskiFastTSNE2020,
  author =       {M. Orli\'nski and N. Jankowski},
  title =        {Fast t-SNE algorithm with forest of balanced LSH trees and hybrid computation of repulsive forces},
  journal =      KBS,
  year =         {2020},
  volume =       {208},
  pages =        {1--16},
  doi =          {https://doi.org/10.1016/j.knosys.2020.106318},
}
```