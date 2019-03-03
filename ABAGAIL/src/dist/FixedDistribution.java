package dist;

import shared.Copyable;
import shared.DataSet;
import shared.Instance;

/**
 * A distribution that does not reestimate
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FixedDistribution extends AbstractDistribution implements Copyable {
    
    /**
     * The distribution
     */
    private Distribution dist;
    
    /**
     * Make a new label distribution
     * @param dist the distribution
     */
    public FixedDistribution(Distribution dist) {
        this.dist = dist;
    }

    /**
     * @see Distribution#probabilityOf(Instance)
     */
    public double p(Instance i) {
        return dist.p(i);
    }

    /**
     * @see Distribution#logLikelihood(Instance)
     */
    public double logp(Instance i) {
        return dist.logp(i);
    }

    /**
     * @see Distribution#generateRandom(Instance)
     */
    public Instance sample(Instance i) {
        return dist.sample(i);
    }
   
    /**
     * @see Distribution#mode(Instance)
     */ 
    public Instance mode(Instance i) {
        return dist.mode(i);
    }

    /**
     * @see Distribution#estimate(DataSet)
     */
    public void estimate(DataSet observations) {
        return;
    }

    /**
     * @see Object#toString()
     */
    public String toString() {
        return "Fixed Distribution " + dist.toString();
    }

    /**
     * @see Copyable#copy()
     */
    public Copyable copy() {
        return new FixedDistribution((Distribution) ((Copyable) dist).copy());
    }
    /**
     * Get the distribution
     * @return returns the distribution
     */
    public Distribution getDistribution() {
        return dist;
    }
    /**
     * Set the distribution
     * @param dist The distribution to set
     */
    public void setDistribution(Distribution dist) {
        this.dist = dist;
    }
}
