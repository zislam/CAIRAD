/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

 /*
 *    CAIRAD.java
 *    Copyright (C) 2020 Michael Furner
 *
 */
package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.Vector;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.SimpleBatchFilter;
import weka.filters.UnsupervisedFilter;

/**
 * <!-- globalinfo-start -->
 * Implements the CAIRAD techique for detecting noisy values in a dataset. Does
 * this with an analysis of coappearance between values. Can output whether or
 * not a record is noisy (i.e. includes 1 or more noisy values), or remove all
 * noisy values and replace them with missing values.
 * <p/>
 * CAIRAD specification from:
 * <p/>
 * Rahman, M. G., Islam, M. Z., Bossomaier, T., & Gao, J. (2012, June). Cairad:
 * A co-appearance based analysis for incorrect records and attribute-values
 * detection. In The 2012 International Joint Conference on Neural Networks
 * (IJCNN) (pp. 1-10). IEEE. Available at
 * http://doi.org/10.1109/IJCNN.2012.6252669
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- options-start -->
 * Valid options are:
 * <p/>
 *
 * <pre> -T
 * coappearanceThreshold - Coappearance Threshold, tau in original paper. </pre>
 *
 * <pre> -L
 * coappearanceScoreThreshold - Coappearance Score Threshold, lambda in original
 * paper. </pre>
 *
 * <pre> -M
 * makeNoisyMissing - Make detected noise into missing values. </pre>
 *
 * <!-- options-end -->
 *
 * @author Michael Furner
 * @version 1.0
 */
public class CAIRAD extends SimpleBatchFilter implements UnsupervisedFilter {

    /**
     * For serialization
     */
    static final long serialVersionUID = -132412310938L;

    /**
     * Coappearance Threshold, tau in original paper.
     */
    private double m_coappearanceThreshold = 0.8;

    /**
     * Coappearance Score Threshold, lambda in original paper.
     */
    private double m_coappearanceScoreThreshold = 0.3;

    /**
     * Make detected noise into missing values
     */
    private boolean m_makeNoisyMissing = true;

    /**
     * Used to store the size of each attribute domain after discretization
     */
    private int[] m_attributeDomainSizes;

    /**
     * Coappearance Matrix used in noise detection
     */
    private CoappearanceMatrix m_CAM;

    /**
     * After NVI process, reflects which attributes in the dataset are noisy
     */
    private int[][] m_noisyAttributeMatrix;

    /**
     * Return a description suitable for displaying in the
     * explorer/experimenter.
     *
     * @return a description suitable for displaying in the
     * explorer/experimenter
     */
    @Override
    public String globalInfo() {
        return "Implements the CAIRAD techique for detecting noisy values in a "
                + "dataset. Does this with an analysis of coappearance between "
                + "values. Can output whether or not a record is noisy (i.e. "
                + "includes 1 or more noisy values), or remove all noisy values "
                + "and replace them with missing values. CAIRAD specification"
                + "from:"
                + "\n"
                + "Rahman, M. G., Islam, M. Z., Bossomaier, T., & Gao, J. "
                + "(2012, June). Cairad: A co-appearance based analysis for "
                + "incorrect records and attribute-values detection. In The 2012"
                + "International Joint Conference on Neural Networks (IJCNN) "
                + "(pp. 1-10). IEEE. \n"
                + "Available at http://doi.org/10.1109/IJCNN.2012.6252669\n"
                + "Valid options are:\n"
                + "-T\n"
                + "coappearanceThreshold - Coappearance Threshold, tau in "
                + "original paper. \n"
                + "\n"
                + "-L\n"
                + "coappearanceScoreThreshold - Coappearance Score Threshold, "
                + "lambda in original paper.\n"
                + "\n"
                + "-M\n"
                + "makeNoisyMissing - Make detected noise into missing values."
                + "\nFor more information see: " + getTechnicalInformation();
    }

    /**
     * Determines the output format based on the input format and returns this.
     *
     * @param inputFormat the input format to base the output format on
     * @return the output format
     * @see #hasImmediateOutputFormat()
     * @see #batchFinished()
     */
    @Override
    protected Instances determineOutputFormat(Instances inputFormat) {
        return inputFormat;
    }

    /**
     * Sets the format of the input instances.
     *
     * @param instanceInfo an Instances object containing the input instance
     * structure (any instances contained in the object are ignored - only the
     * structure is required).
     * @return true if the outputFormat may be collected immediately
     * @throws Exception if the input format can't be set successfully
     */
    @Override
    public boolean setInputFormat(Instances instanceInfo)
            throws Exception {

        super.setInputFormat(instanceInfo);
        setOutputFormat(instanceInfo);
        return true;
    }

    /**
     * Input an instance for filtering. Filter requires all training instances
     * be read before producing output.
     *
     * @param instance the input instance
     * @return true if the filtered instance may now be collected with output().
     * @throws IllegalStateException if no input structure has been defined
     */
    @Override
    public boolean input(Instance instance) {

        if (getInputFormat() == null) {
            throw new IllegalStateException("No input instance format defined");
        }
        if (m_NewBatch) {
            resetQueue();
            m_NewBatch = false;
        }
        if (isFirstBatchDone()) {
            push(instance);
            return true;
        } else {
            bufferInput(instance);
            return false;
        }
    }

    /**
     * Perform CAIRAD on given dataset
     *
     * @param input - dataset to process
     * @return Dataset with either detected noisy values removed, or with noisy
     * records indicated with a new boolean attribute
     * @throws Exception
     */
    @Override
    protected Instances process(Instances input) throws Exception {

        this.setInputFormat(input);
        /*Step 1: Generalise numerical attributes in copy of dataset */
        Instances generalisedDataset = new Instances(input);
        Instances originalDataset = new Instances(input);

        Discretize discretizer = new Discretize();
        m_attributeDomainSizes = new int[originalDataset.numAttributes()];

        for (int i = 1; i <= generalisedDataset.numAttributes(); i++) {

            if (generalisedDataset.attribute(i - 1).isDate()) {

                //work out the number of bins from the range
                double range = generalisedDataset.attributeStats(i - 1).distinctCount;

                int numBins = (int) Math.round(Math.sqrt(range));

                //set up and run the discretiser
                discretizer.setAttributeIndices("" + i);
                discretizer.setInputFormat(generalisedDataset);
                discretizer.setBins(numBins);
                m_attributeDomainSizes[i - 1] = numBins;

                generalisedDataset = Filter.useFilter(generalisedDataset, discretizer);
            } else if (generalisedDataset.attribute(i - 1).isNumeric()) {

                //work out the number of bins from the range
                double range = generalisedDataset.attributeStats(i - 1).numericStats.max
                        - generalisedDataset.attributeStats(i - 1).numericStats.min;

                int numBins = (int) Math.round(Math.sqrt(range));

                //set up and run the discretiser
                discretizer.setAttributeIndices("" + i);
                discretizer.setInputFormat(generalisedDataset);
                discretizer.setBins(numBins);
                m_attributeDomainSizes[i - 1] = numBins;

                generalisedDataset = Filter.useFilter(generalisedDataset, discretizer);

            }//end if numeric
            else if (generalisedDataset.attribute(i - 1).isString()) {
                StringToNominal stn = new StringToNominal();
                stn.setAttributeRange("" + i);
                stn.setInputFormat(generalisedDataset);
                generalisedDataset = Filter.useFilter(generalisedDataset, stn);

//                MergeInfrequentNominalValues minv = new MergeInfrequentNominalValues();
//                minv.setAttributeIndices(""+i);
//                minv.setInputFormat(generalisedDataset);
//                generalisedDataset = Filter.useFilter(generalisedDataset, minv);
                m_attributeDomainSizes[i - 1] = generalisedDataset.attributeStats(i - 1).nominalCounts.length;

            } else {
                m_attributeDomainSizes[i - 1] = generalisedDataset.attributeStats(i - 1).nominalCounts.length;
            }

        } //end generalisation loop

        /*Step 2: Generate a coappearance matrix on generalised dataset */
        m_CAM = new CoappearanceMatrix(generalisedDataset);

        /*Step 3: Identify noisy values */
        //create noisy attribute matrix Q
        m_noisyAttributeMatrix = new int[generalisedDataset.numInstances()][generalisedDataset.numAttributes()];
        boolean[] isNoisy = new boolean[generalisedDataset.numInstances()];
        for (int i = 0; i < generalisedDataset.numInstances(); i++) {
            isNoisy[i] = NVI(generalisedDataset.get(i), i);
        }

        /*Step 4: Produce dataset with all clean records and dataset with all
                  noisy records */
        //this step is unnecessary for this implementation
        /*Step 5: Package it up to return the final result */
        //If we're replacing all of the noisy values with missing values for
        //later imputation
        if (m_makeNoisyMissing) {
            for (int i = 0; i < originalDataset.numInstances(); i++) {
                for (int j = 0; j < originalDataset.numAttributes(); j++) {

                    if (m_noisyAttributeMatrix[i][j] == 1) { //if noisy
                        originalDataset.instance(i).setValue(j, Utils.missingValue());
                    }

                }
            }
            this.setOutputFormat(originalDataset);
        } else { //otherwise just create an indicator variable for whether or not the records are noisy
            ArrayList values = new ArrayList();
            values.add("False");
            values.add("True");
            originalDataset.insertAttributeAt(new Attribute("Noisy", values), 0);

            for (int i = 0; i < originalDataset.numInstances(); i++) {
                if (isNoisy[i]) {
                    originalDataset.instance(i).setValue(0, 1);
                } else {
                    originalDataset.instance(i).setValue(0, 0);
                }
            }
            this.setOutputFormat(originalDataset);
        }

        return originalDataset;

    }

    /**
     * Return the noisy attribute matrix (Q in original paper)
     *
     * @return the noisy attribute matrix
     */
    public int[][] getNoisyAttributeMatrix() {
        return m_noisyAttributeMatrix;
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String coappearanceThresholdTipText() {
        return "Coappearance Threshold, tau in original paper";
    }

    /**
     * Return Coappearance Threshold, tau in original paper
     *
     * @return Coappearance Threshold, tau in original paper
     */
    public double getCoappearanceThreshold() {
        return m_coappearanceThreshold;
    }

    /**
     * Set Coappearance Threshold, tau in original paper
     *
     * @param m_coappearanceThreshold - new coappearance threshold
     */
    public void setCoappearanceThreshold(double m_coappearanceThreshold) {
        this.m_coappearanceThreshold = m_coappearanceThreshold;
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String coappearanceScoreThresholdTipText() {
        return "Coappearance Score Threshold, lambda in original paper";
    }

    /**
     * Returns Coappearance Score Threshold, lambda in original paper
     *
     * @return Coappearance Score Threshold
     */
    public double getCoappearanceScoreThreshold() {
        return m_coappearanceScoreThreshold;
    }

    /**
     * Set Coappearance Score Threshold, lambda in original paper
     *
     * @param m_coappearanceScoreThreshold - new coappearance score threshold
     */
    public void setCoappearanceScoreThreshold(double m_coappearanceScoreThreshold) {
        this.m_coappearanceScoreThreshold = m_coappearanceScoreThreshold;
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String makeNoisyMissingTipText() {
        return "Make detected noise into missing values";
    }

    /**
     * Return whether or not to make detected noisy values into missing values
     *
     * @return m_makeNoisyMissing
     */
    public boolean getMakeNoisyMissing() {
        return m_makeNoisyMissing;
    }

    /**
     * Set whether or not to make detected noisy values into missing values
     *
     * @param makeNoisyMissing whether or not to make detected noisy values into
     * missing values
     */
    public void setMakeNoisyMissing(boolean makeNoisyMissing) {
        this.m_makeNoisyMissing = makeNoisyMissing;
    }

    /**
     * Parses a given list of options.
     * <p/>
     *
     * <!-- options-start -->
     * Valid options are:
     * <p/>
     * <
     * pre> -T
     * coappearanceThreshold - Coappearance Threshold, tau in original paper.
     * </pre>
     *
     * <pre> -L
     * coappearanceScoreThreshold - Coappearance Score Threshold, lambda in original
     * paper. </pre>
     *
     * <pre> -M
     * makeNoisyMissing - Make detected noise into missing values. </pre>
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        String optionString;

        super.setOptions(options);

        // set coappearance threshold - T for Tau
        optionString = Utils.getOption('T', options);
        if (optionString.length() != 0) {
            double tau = Double.parseDouble(optionString);
            setCoappearanceThreshold(tau);
            if ((tau <= 0) || (tau > 1)) {
                throw new Exception(
                        "Coappearance Threshold must be > 0 and <= 1"
                );
            }
        }

        // set coappearance score threshold - L for Lambda
        optionString = Utils.getOption('L', options);
        if (optionString.length() != 0) {
            double lambda = Double.parseDouble(optionString);
            setCoappearanceScoreThreshold(lambda);
            if ((lambda <= 0) || (lambda > 1)) {
                throw new Exception(
                        "Coappearance Score Threshold must be > 0 and <= 1"
                );
            }

        }

        //set whether or not to replace noisy values with missing values
        setMakeNoisyMissing(Utils.getFlag("M", options));

    }

    /**
     * Gets the current settings of EMImputation
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    public String[] getOptions() {

        Vector<String> result = new Vector<String>();

        result.add("-T");
        result.add("" + getCoappearanceThreshold());

        result.add("-L");
        result.add("" + getCoappearanceScoreThreshold());

        if (getMakeNoisyMissing()) {
            result.add("-M");
        }

        return result.toArray(new String[result.size()]);

    }

    /**
     * Perform noisy value identification on a record using the coappearance
     * matrix. Works out expected coappearance for particular values, and
     * indicates that a value could be noisy based on the actual coappearance.
     *
     * @param theRecord - record to perform noisy value identification on.
     * @param index - index of theRecord in the dataset
     * @return
     */
    private boolean NVI(Instance theRecord, int index) {
        boolean isNoisy = false;

        int[] totalScores = new int[theRecord.numAttributes()];

        for (int j = 0; j < theRecord.numAttributes() - 1; j++) {

            for (int k = j + 1; k < theRecord.numAttributes(); k++) {

                int x = (int) theRecord.value(j);
                int y = (int) theRecord.value(k);

                //get frequencies of these values
                double xf = m_CAM.valueAppearances[j][x];
                double yf = m_CAM.valueAppearances[k][y];

                //get attribute domain sizes
                double Aj = m_attributeDomainSizes[j];
                double Ak = m_attributeDomainSizes[k];

                //get expected coappearance
                double Exy = (xf / Ak) * m_coappearanceThreshold;
                double Eyx = (yf / Aj) * m_coappearanceThreshold;

                //get actual coappearances
                int Cxy = m_CAM.matrix[j][x][k][y];

                if (Cxy < Exy && Cxy < Eyx) {
                    totalScores[j] += 2;
                    totalScores[k] += 2;
                } else if (Cxy > Exy && Cxy > Eyx) { //these are pointless statements but make it more readable
                    totalScores[j] += 0;
                    totalScores[k] += 0;
                } else {
                    totalScores[j] += 1;
                    totalScores[k] += 1;
                }

            } //end attr k loop

        } //end attr j loop

        for (int j = 0; j < theRecord.numAttributes(); j++) {

            if (totalScores[j] / ((theRecord.numAttributes() - 1.0) * 2.0) > m_coappearanceScoreThreshold) {
                isNoisy = true;
                m_noisyAttributeMatrix[index][j] = 1;
            }

        }

        return isNoisy;
    }

    /**
     * Class for a coapparance matrix.
     */
    final class CoappearanceMatrix {

        /**
         * The actual CAM. Dimensions are:
         * [attribute_i][att_i:value_a][attribute_j][att_j:value_b] the values
         * count the number of coappearances between value a of attribute i and
         * value b of attribute j.
         */
        public int[][][][] matrix;

        /**
         * Counter of how many times value_a of attribute_i appears. Dimensions:
         * [attribute_i][att_i:value_a]
         */
        public int[][] valueAppearances;

        /**
         * Initialise CAM on dataset
         *
         * @param dataset
         */
        CoappearanceMatrix(Instances dataset) {

            matrix = new int[dataset.numAttributes()][][][];
            valueAppearances = new int[dataset.numAttributes()][];
            constructCAM(dataset);

        }

        /**
         * Build the coappearance matrix on ds.
         *
         * @param ds
         */
        public void constructCAM(Instances ds) {

            //initialise the arrays to hold coappearances
            for (int i = 0; i < ds.numAttributes(); i++) {

                //set up attribute i's set of attribute values
                matrix[i] = new int[ds.attributeStats(i).nominalCounts.length][][];
                valueAppearances[i] = new int[ds.attributeStats(i).nominalCounts.length];

                for (int j = 0; j < matrix[i].length; j++) {

                    //set up attribute i value j's counts for each attribute
                    matrix[i][j] = new int[ds.numAttributes()][];

                    for (int k = 0; k < matrix[i][j].length; k++) {

                        matrix[i][j][k] = new int[ds.attributeStats(k).nominalCounts.length];

                    } //end attribute k loop

                } //end attribute i value j loop

            } //end attribute i loop

            //iterate over the dataset
            for (int i = 0; i < ds.numInstances(); i++) {

                Instance theRecord = ds.instance(i);

                for (int attrOneIndex = 0; attrOneIndex < ds.numAttributes(); attrOneIndex++) {

                    double attrOneValue = theRecord.value(attrOneIndex);
                    valueAppearances[attrOneIndex][(int) attrOneValue]++;

                    for (int attrTwoIndex = 0; attrTwoIndex < ds.numAttributes(); attrTwoIndex++) {

                        if (attrOneIndex != attrTwoIndex) { //obviously don't do the second one
                            double attrTwoValue = theRecord.value(attrTwoIndex);

                            matrix[attrOneIndex][(int) attrOneValue][attrTwoIndex][(int) attrTwoValue]++;

                        }

                    } //end of second attr loop

                } //end of attr loop

            }

        }

    }

    /**
     * Returns the Capabilities of this filter.
     *
     * @return the capabilities of this object
     * @see Capabilities
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.STRING_ATTRIBUTES);
        result.enable(Capabilities.Capability.DATE_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.enable(Capabilities.Capability.NO_CLASS);
        result.enable(Capabilities.Capability.BINARY_CLASS);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);

        return result;
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(TechnicalInformation.Type.CONFERENCE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Rahman, M. G., Islam, M. Z., Bossomaier, T., & Gao, J.");
        result.setValue(TechnicalInformation.Field.YEAR, "2012");
        result.setValue(TechnicalInformation.Field.TITLE, "Cairad: A co-appearance based analysis for incorrect records and attribute-values detection");
        result.setValue(TechnicalInformation.Field.BOOKTITLE, "The 2012 International Joint Conference on Neural Networks (IJCNN)");
        result.setValue(TechnicalInformation.Field.PUBLISHER, "IEEE");
        result.setValue(TechnicalInformation.Field.ADDRESS, "Brisbane, QLD, Australia");
        result.setValue(TechnicalInformation.Field.URL, "https://s3.amazonaws.com/academia.edu.documents/54151630/CAIRAD_A_Co-appearance_based_Analysis_fo20170815-24324-1ozdcxp.pdf?response-content-disposition=inline%3B%20filename%3DCAIRAD_A_co-appearance_based_analysis_fo.pdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=ASIATUSBJ6BAFPOD4P74%2F20200501%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200501T222400Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEP7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIHY9eZgyczDVgJ3uKG5%2F45vnn9FGmmeLRjwifAIoqXqeAiEAkrCNUXC15Pq93%2FjSqtO8YX29gIgpS78loVbS4Zt7a88qtAMINhAAGgwyNTAzMTg4MTEyMDAiDKFvYvLMAVivg3H1pyqRA1uSQpdnuBx9FXVm%2FUF024oQ1LZViiYLGQxNg7vPwrTrKRiT7wp1pU%2Bg%2FtK9CDymeVwE3VBA8GsHywnoOl8wLyFCyd0d6BVkhNAYr5JK6h2fHCuws0WncmLWRdOhPgDj70LlapxPKLlrNh5byOSEIAwmk4O8e6VAZATtKmUytDFoWaJzd%2B8aZP7sZn3sxYnE%2BFZeYMrzJQF1HXU%2FE5TEDfLdMNwT98TRvPLKlZR1zSaS0Q%2FR7l%2BjoSZat31Bd9V6JXy15aa37mm3k2LZKPJSoJ0Cc0dhEmBoGeCu0VtWzcqTWmk9Mx9OmgyxQwhZvX2DFexJM11KaWCA4%2Bd2l6zk63gL3665f4Mzb0qmaPVQJMjbkgzOxA7xmve75XGMDhtpjDOha5k7xjf8zp316RLYlSlkG5xDCBJ8Q%2BQJUtX1NGz%2BF8SJhl5QIzaoyAKJje%2BnL8GIbPCV%2Bz70brafTzWVaupjeMkoY5dizgATXSa5FD7fcJ5q6FioVu4ZPXGCwXL5mhEpQTesMjinsAXAGVRZCSKtMNGgsvUFOusB8n9elUoJkipHjvs%2FXtwNnYUuemFRiU9PoLbqHFE6l6fX9hv%2Bdw4tklwkZdEVRINODl1998QIJann3P2Ih9Bg%2FE%2F5p1dZW0%2FJkqFDzMDu6%2FcDBKcTBy4NCptAHgXhGDNpDAKwIJFcLTtnTsFbHr4vxDdkmYGfnUO0oAhPdAfB%2BbiHRJfAQYOmCXb%2BgBh5bRtY%2BsIcZ5me8bYCy%2FrAtGyZ87E3I17tlTPlK9C5fjew8Vbb21vzXzywu0HM2cULqbm6lswcBNJ7D1GLGb1bitp8HCRflvVXicv4KyU2dgYKhwF4ia%2BO6mVLI7JTvA%3D%3D&X-Amz-Signature=227c7811b5f060706ba12844b3ca886a2756ad74dadf7397fad4877ca7f7d417");

        return result;

    }

    /**
     * Main method for testing this class.
     *
     * @param argv should contain arguments to the filter: use -h for help
     */
    public static void main(String[] argv) {
        runFilter(new CAIRAD(), argv);
    }

}
