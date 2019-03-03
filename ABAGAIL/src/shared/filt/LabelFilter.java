package shared.filt;

import shared.DataSet;

/**
 * A filter that applies a filter to the label data set
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class LabelFilter implements ReversibleFilter {
    /**
     * The filter to apply
     */
    private DataSetFilter filter;

    /**
     * @see DataSetFilter#filter(DataSet)
     */
    public void filter(DataSet dataSet) {
        filter.filter(dataSet.getLabelDataSet());
    }

    /**
     * @see ReversibleFilter#reverse(DataSet)
     */
    public void reverse(DataSet set) {
        ((ReversibleFilter) filter).reverse(set.getLabelDataSet());
    }

}
