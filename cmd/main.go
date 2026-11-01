package main

import (
	"fmt"

	"github.com/jrbarbati/neural-net/internal/activation"
	"github.com/jrbarbati/neural-net/internal/layer"
	"github.com/jrbarbati/neural-net/internal/network"
	"github.com/jrbarbati/neural-net/internal/visualize"
	"github.com/moverest/mnist"
)

func main() {
	training, testing, err := mnist.Load("/home/jrbarbati/dev/neural-net/mnist")

	if err != nil {
		panic(err)
	}

	for _, img := range training.Images {
		fmt.Println(len(*img))
		break
	}

	n := network.New([]layer.Config{
		{InputLen: 784, Size: 16, ActivationFn: activation.Sigmoid},
		{InputLen: 16, Size: 16, ActivationFn: activation.Sigmoid},
		{InputLen: 16, Size: 10, ActivationFn: activation.Sigmoid},
	})

	rec, err := visualize.Wrap(n, visualize.Options{SampleEvery: 100})
	if err != nil {
		panic(err)
	}

	go func() {
		if err := rec.ListenAndServe("localhost:8081"); err != nil {
			panic(err)
		}
	}()

	fmt.Println("viewer: http://localhost:8081 — press Enter to start training")
	fmt.Scanln() // gives you time to open the page so you can watch it live

	train(rec, training)

	if err := rec.SaveHTML("network.html"); err != nil { // replay you can open later without the server
		panic(err)
	}

	fmt.Println("training done; viewer still at http://localhost:8081 — Ctrl+C to exit")

	testImg := make([][]float64, 784)

	for j := range testImg {
		testImg[j] = []float64{float64(testing.Images[0][j])}
	}

	_, err = rec.Categorize(testImg)
	if err != nil {
		fmt.Println(err)
	}

	select {}

	//train(n, training)
	//
	//testImg := make([][]float64, 784)
	//
	//for i := range testImg {
	//	testImg[i] = []float64{float64(testing.Images[0][i])}
	//}
	//
	//categorize, err := n.Categorize(testImg)
	//
	//if err != nil {
	//	fmt.Println(err.Error())
	//	return
	//}
	//
	//fmt.Println(categorize)
}

func train(n visualize.Model, trainingSet *mnist.Set) {
	for i, img := range trainingSet.Images {
		trainImg := make([][]float64, 784)

		for j := range trainImg {
			trainImg[j] = []float64{float64(img[j])}
		}

		err := n.Train(trainImg)

		if err != nil {
			fmt.Printf("Error while training on image %v: %v\n", i, err.Error())
		}
	}
}
