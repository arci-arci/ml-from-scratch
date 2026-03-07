package knn

import (
	"math"
	"ml/common"
	"os"
	"path"
	"sort"
	"strings"
)

type Point struct {
	Document *common.BoW
	Class    string
}

type KNNModel struct {
	Points     []NormalizedDocument
	Vocabulary *common.Vocabulary
	Size       int
	df         *DocumentFrequency
}

type Neighbor struct {
	Distance     float64
	Index        int
	Class        string
	DocumentName string
}

type DocumentData struct {
	DocumentName string
	Bow          *common.BoW
	Class        string
}

type NormalizedDocument struct {
	DocumentName string
	WBow         *common.WeightedBoW
	Class        string
}

type DocumentFrequency = map[string]int

func Train(folders []string, classes []string) KNNModel {
	v := common.Vocabulary{}
	db, df := normalize(folders, classes)
	defineVocabulary(db, &v)

	return KNNModel{
		Points:     db,
		Vocabulary: &v,
		Size:       len(db),
		df:         &df,
	}
}

func Fit(model *KNNModel, p *common.BoW, k int) string {
	if k <= 0 {
		panic("k parameter must be greater than 0")
	}

	neighbors := []Neighbor{}
	target := getWeithedBoW(p, model.df, model.Size)

	for index, q := range model.Points {
		d := cosineSimilarity(&target, q.WBow, model.Vocabulary)
		neighbor := Neighbor{
			Distance: d, Index: index,
			Class: q.Class, DocumentName: q.DocumentName,
		}

		neighbors = append(neighbors, neighbor)
	}

	sort.Slice(neighbors, func(i int, j int) bool {
		return neighbors[i].Distance > neighbors[j].Distance
	})

	return vote(neighbors[:k])
}

func defineVocabulary(db []NormalizedDocument, v *common.Vocabulary) {
	for _, document := range db {
		common.CreateWeightedVocabulary(document.WBow, v)
	}
}

func normalize(folders []string, classes []string) ([]NormalizedDocument, DocumentFrequency) {
	size, err := common.GetCollectionSize(folders, classes)
	df := DocumentFrequency{}
	documentsData := make([]DocumentData, 0, size)
	normalizedDocuments := make([]NormalizedDocument, 0, size)

	if err != nil {
		panic(err)
	}

	for _, folder := range folders {
		for _, class := range classes {
			documents, err := os.ReadDir(path.Join(folder, class))
			if err != nil {
				panic(err)
			}

			for _, document := range documents {
				if strings.ToLower(document.Name()) == "summary.txt" {
					continue
				}

				bow := common.BoW{}
				common.ReadClassDocument(folder, class, document.Name(), &bow)

				if len(bow) < common.AVG_TOKEN_AMOUNT {
					continue
				}

				addToDocFrequency(&df, &bow)
				documentsData = append(documentsData, DocumentData{DocumentName: document.Name(), Bow: &bow, Class: class})
			}
		}
	}

	for _, info := range documentsData {
		weightedBoW := getWeithedBoW(info.Bow, &df, size)
		nDoc := NormalizedDocument{
			DocumentName: info.DocumentName,
			Class:        info.Class,
			WBow:         &weightedBoW,
		}

		normalizedDocuments = append(normalizedDocuments, nDoc)
	}

	return normalizedDocuments, df
}

func calculateTermFrequency(bow *common.BoW) common.WeightedBoW {
	var totalFrequencies int64
	tfBoW := common.WeightedBoW{}

	for token := range *bow {
		totalFrequencies += (*bow)[token]
	}

	for token := range *bow {
		tfBoW[token] = float64((*bow)[token]) / float64(totalFrequencies)
	}

	return tfBoW
}

func getWeithedBoW(bow *common.BoW, df *DocumentFrequency, size int) common.WeightedBoW {
	weightedBoW := common.WeightedBoW{}
	tfBoW := calculateTermFrequency(bow)

	for token, tf := range tfBoW {
		// In case a term is not present in a document, I add 1 on both of the term
		idf := math.Log10(float64(1+size) / float64(1+(*df)[token]))
		weightedBoW[token] = tf * idf
	}

	return weightedBoW
}

func addToDocFrequency(df *DocumentFrequency, bow *common.BoW) {
	for token := range *bow {
		numDocs, ok := (*df)[token]

		if ok {
			(*df)[token] = numDocs + 1
		} else {
			(*df)[token] = 1
		}
	}
}

func vote(neighbors []Neighbor) string {
	classes := []string{"ham", "spam"}
	foundClass := classes[0]
	maxFrequency := -1

	for _, class := range classes {
		frequency := 0

		for _, n := range neighbors {
			if class == n.Class {
				frequency += 1
			}
		}

		if frequency > maxFrequency {
			foundClass = class
			maxFrequency = frequency
		}
	}

	return foundClass
}

func cosineSimilarity(target *common.WeightedBoW, q *common.WeightedBoW, v *common.Vocabulary) float64 {
	targetLen := magnitude(target, v)
	pointLen := magnitude(q, v)
	product := dot(target, q, v)
	distance := product / (targetLen * pointLen)

	return distance
}

func dot(target *common.WeightedBoW, q *common.WeightedBoW, v *common.Vocabulary) float64 {
	var sum float64

	for token := range *v {
		targetV := (*target)[token]
		pointV := (*q)[token]
		sum = sum + (targetV * pointV)
	}

	return sum
}

func magnitude(vector *common.WeightedBoW, v *common.Vocabulary) float64 {
	var sum float64

	for token := range *v {
		val := (*vector)[token]
		sum += math.Pow(val, 2)
	}

	return math.Sqrt(sum)
}
