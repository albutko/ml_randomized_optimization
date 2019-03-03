package exp.tests;

import util.linalg.Vector;
import func.nn.NeuralNetwork;
import opt.EvaluationFunction;
import shared.DataSet;
import shared.Instance;

/**
 * An evaluation function that uses a neural network
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class NeuralNetworkEvaluationFunctionRecall implements EvaluationFunction {
    /**
     * The network
     */
    private NeuralNetwork network;
    /**
     * The examples
     */
    private DataSet examples;
    /**
     * The error measure
     */
    private MyRecallTestMetric metric;

    /**
     * Make a new neural network evaluation function
     * @param network the network
     * @param examples the examples
     * @param metric
     */
    public NeuralNetworkEvaluationFunctionRecall(NeuralNetwork network,
                                           DataSet examples, MyRecallTestMetric metric) {
        this.network = network;
        this.examples = examples;
        this.metric = metric;
    }

    /**
     * @see opt.OptimizationProblem#value(opt.OptimizationData)
     */
    public double value(Instance d) {
        // set the links
        Vector weights = d.getData();
        network.setWeights(weights);

        for (int i = 0; i < examples.size(); i++) {
            network.setInputValues(examples.get(i).getData());
            network.run();
            int act = new Instance(network.getOutputValues()).getDiscrete();
            metric.addResult(new Instance(examples.get(i).getData(), new Instance(examples.get(i).getLabel().getDiscrete())), new Instance(examples.get(i).getData(), new Instance(act)));
        }
        // the fitness recall score
        double recall = metric.getPctRecall();
        this.metric = new MyRecallTestMetric();
        return recall;
    }

}
