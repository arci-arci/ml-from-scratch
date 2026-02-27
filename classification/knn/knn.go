package knn

import (
	"math"
	"ml/common"
	"os"
	"path"
	"sort"
)

type Point struct {
	Document *common.BoW
	Class    string
}

type KNNModel struct {
	Points []Point
	V      *common.Vocabulary
}

type NearestPoints struct {
	Distance float64
	Index    int
	Class    string
}

const initialCapacity int = 10_000

func Train(folders []string) KNNModel {
	db := make([]Point, 0, initialCapacity)
	v := common.Vocabulary{}
	classes := []string{"ham", "spam"}

	for _, folder := range folders {
		for _, class := range classes {
			documents, err := os.ReadDir(path.Join(folder, class))
			if err != nil {
				panic(err)
			}

			for _, document := range documents {
				bow := common.BoW{}
				common.ReadClassDocument(folder, class, document.Name(), &bow)
				db = append(db, Point{Document: &bow, Class: class})
				common.CreateVocabulary(&bow, &v)
			}
		}
	}

	return KNNModel{
		Points: db,
		V:      &v,
	}
}

func Fit(model KNNModel, p *common.BoW, k int) []NearestPoints {
	distances := []NearestPoints{}

	if k <= 0 {
		panic("k parameter must be greater than 0")
	}

	for index, q := range model.Points {
		d := euclideanDistance(p, q.Document, model.V)
		distances = append(distances, NearestPoints{Distance: d, Index: index, Class: q.Class})
	}

	sort.Slice(distances, func(i int, j int) bool {
		return distances[i].Distance < distances[j].Distance
	})

	return distances[:k]
}

func euclideanDistance(p *common.BoW, q *common.BoW, v *common.Vocabulary) float64 {
	var distance float64

	for token := range *v {
		x := (*p)[token]
		y := (*q)[token]
		squareValue := math.Pow(float64(x-y), 2)
		distance += squareValue
	}

	return math.Sqrt(distance)
}
