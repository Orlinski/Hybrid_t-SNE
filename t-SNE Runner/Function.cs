using System;
using NCalc;

namespace tSNE_Runner
{
    class Function
    {
        private Expression expression;

        public static Func<int, int, double> MakeFunction(string definition)
        {
            Function instance = new Function(definition);
            return instance.Call;
        }

        private Function(string definition)
        {
             expression = new Expression(definition);
        }

        private double Call(int Iteration, int Iterations)
        {
            expression.Parameters["Iteration"] = Iteration;
            expression.Parameters["Iterations"] = Iterations;
            return Convert.ToDouble(expression.Evaluate());
        }
    }
}
