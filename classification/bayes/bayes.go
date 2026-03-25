package bayes

import (
	"math"
	"ml/common"
)

type probabilities struct {
	ham  float64
	spam float64
}

type ClassificationOptions struct {
	HamProb  float64
	SpamProb float64
	Model    *BayesModel
	Doc      *common.BoW
}

type BayesOptions struct {
	HamBoW     *common.BoW
	SpamBoW    *common.BoW
	Vocabulary *common.Vocabulary
}

type ClassificationResult struct {
	Ham  float64
	Spam float64
}

type BayesModel = map[string]probabilities

func Train(options BayesOptions) BayesModel {
	classifier := BayesModel{}
	termsForHam := common.CalcualteTermsAmount(options.HamBoW)
	termsForSpam := common.CalcualteTermsAmount(options.SpamBoW)
	vSize := len(*options.Vocabulary)

	for token := range *options.Vocabulary {
		// Probability calculated based on Laplace Correction
		probForHam := float64((*options.HamBoW)[token]+1) / float64(termsForHam+int64(vSize))
		probForSpam := float64((*options.SpamBoW)[token]+1) / float64(termsForSpam+int64(vSize))
		tProb := probabilities{
			ham:  probForHam,
			spam: probForSpam,
		}

		classifier[token] = tProb
	}

	return classifier
}

func Fit(options ClassificationOptions) ClassificationResult {
	docInHam := 0.0
	docInSpam := 0.0

	for token := range *options.Doc {
		tProb := (*options.Model)[token]

		// Skipping rare tokens
		if tProb.ham == 0 || tProb.spam == 0 {
			continue
		}

		docInHam += math.Log(tProb.ham)
		docInSpam += math.Log(tProb.spam)
	}

	return ClassificationResult{
		Ham:  docInHam + math.Log(options.HamProb),
		Spam: docInSpam + math.Log(options.SpamProb),
	}
}
