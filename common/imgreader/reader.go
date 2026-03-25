package imgreader

import (
	"image"
	"image/png"
	"io"
	"os"
)

type Pixel struct {
	R int
	G int
	B int
}

const conversionUnit uint32 = 257

func GetPixelFromImage(path string) ([][]Pixel, error) {
	image.RegisterFormat("png", "png", png.Decode, png.DecodeConfig)
	file, err := os.Open(path)

	if err != nil {
		return nil, err
	}

	defer file.Close()
	pixels, err := getPixels(file)

	if err != nil {
		return nil, err
	}

	return pixels, nil
}

func getPixels(file io.Reader) ([][]Pixel, error) {
	img, _, err := image.Decode(file)

	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y
	var pixels [][]Pixel

	for y := range height {
		var row []Pixel

		for x := range width {
			r, g, b, _ := img.At(x, y).RGBA()
			row = append(row, rgbaToPixel(r, g, b))
		}

		pixels = append(pixels, row)
	}

	return pixels, nil
}

func rgbaToPixel(r uint32, g uint32, b uint32) Pixel {
	// Because we are dealing with 16-bit alpha-premultiplied color
	// in the range [0, 65535], we convert each compoent's value to a
	// 8-bit alpha-premultiplied color in the range [0, 255]
	// by dividing each component by 0x101 (257)
	//
	// https://go.dev/blog/image#colors-and-color-models

	return Pixel{
		R: int(r / conversionUnit),
		G: int(g / conversionUnit),
		B: int(b / conversionUnit),
	}
}
