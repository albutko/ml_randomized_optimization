package dist;

import shared.DataSet;
import shared.Instance;
import util.ABAGAILArrays;

/**
 * A distribution of all of the permutations
 * of a set size.
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class DiscretePermutationDistribution extends AbstractDistribution {
    /**
     * The size of the data
     */
    private int n;
    
    /**
     * The probability
     */
    private double p;
    
    /**
     * Make a new discrete permutation distribution
     * @param n the size of the data
     */
    public DiscretePermutationDistribution(int n) {
        this.n = n;
        p = n;
        for (int i = n - 1; i >= 1; i--) {
            p *= i;
        }
        p = 1 / p;
    }

    /**
     * @see Distribution#probabilityOf(Instance)
     */
    public double p(Instance i) {
        return p;
    }

    /**
     * @see Distribution#generateRandom(Instance)
     */
    public Instance sample(Instance ignored) {
        double[] d  = ABAGAILArrays.dindices(n);
        ABAGAILArrays.permute(d);
        return new Instance(d);
    }

    /**
     * @see Distribution#generateMostLikely(Instance)
     */
    public Instance mode(Instance ignored) {
        return sample(ignored);
    }

    /**
     * @see Distribution#estimate(DataSet)
     */
    public void estimate(DataSet observations) {
        return;
    }
}
