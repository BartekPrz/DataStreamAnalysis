package moa.classifiers.meta;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.evaluation.WindowAUCImbalancedPerformanceEvaluator;
import moa.options.ClassOption;

public class HybridNeighbourhoodOnlineBagging extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train", Classifier.class, "trees.HoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag", 10, 1, Integer.MAX_VALUE);

    public IntOption numberOfNeighboursOption = new IntOption("neighboursCount", 'k',
            "Number of neighbours taken into account during analysis", 5, 3, Integer.MAX_VALUE);

    public IntOption windowSizeOption = new IntOption("windowSize", 'w',
            "The window size used to analyze the neighbourhood of incoming example", 1000, 1, Integer.MAX_VALUE);

    public FloatOption psiCoefficient = new FloatOption("psi", 'p',
            "Additional coefficient for calculating safe level", 1, 0.5, 3);

    protected NOOBForHybridExtension noobClassifier;

    protected NUOBForHybridExtension nuobClassifier;

    protected WindowAUCImbalancedPerformanceEvaluator noobEvaluator;

    protected WindowAUCImbalancedPerformanceEvaluator nuobEvaluator;

    @Override
    public String getPurposeString() {
        return "Hybrid neighbourhood online bagging for imbalanced data";
    }

    @Override
    public void resetLearningImpl() {
        this.noobClassifier = new NOOBForHybridExtension(this.windowSizeOption.getValue(), this.numberOfNeighboursOption.getValue(), this.psiCoefficient.getValue(), baseLearnerOption, ensembleSizeOption);
        this.nuobClassifier = new NUOBForHybridExtension(this.windowSizeOption.getValue(), this.numberOfNeighboursOption.getValue(), this.psiCoefficient.getValue(), baseLearnerOption, ensembleSizeOption);

        this.noobEvaluator = new WindowAUCImbalancedPerformanceEvaluator();
        this.nuobEvaluator = new WindowAUCImbalancedPerformanceEvaluator();

        this.noobClassifier.prepareForUse();
        this.nuobClassifier.prepareForUse();

        this.noobClassifier.resetLearningImpl();
        this.nuobClassifier.resetLearningImpl();

        this.noobEvaluator.reset(2);
        this.nuobEvaluator.reset(2);

        this.noobEvaluator.widthOption.setValue(1000);
        this.nuobEvaluator.widthOption.setValue(1000);
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        initVariables();

        InstanceExample example = new InstanceExample(inst);
        noobEvaluator.addResult(example, noobClassifier.getVotesForInstance(inst));
        nuobEvaluator.addResult(example, nuobClassifier.getVotesForInstance(inst));

        noobClassifier.trainOnInstanceImpl(inst);
        nuobClassifier.trainOnInstanceImpl(inst);
    }

    private void initVariables() {
        if (this.noobClassifier.windowInstances == null) {
            this.noobClassifier.setModelContext(getModelContext());
        }
        if (this.nuobClassifier.windowInstances == null) {
            this.nuobClassifier.setModelContext(getModelContext());
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return noobEvaluator.getAucEstimator().getGMean() > nuobEvaluator.getAucEstimator().getGMean() ?
                noobClassifier.getVotesForInstance(inst) :
                nuobClassifier.getVotesForInstance(inst);
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }
}
