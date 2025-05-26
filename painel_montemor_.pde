int cols = 22;         // number of columns
int rows = 14;         // adjust to match your data
int tileWidth = 2200 / cols;
int tileHeight = tileWidth;  // square tiles

PImage[][] images;
PImage[][] binaryImages;  // to store converted images

String letters = "ABCDFHIJKLMNOPQO";

void setup() {
  size(2200, 1400);
  images = new PImage[rows][cols];
  binaryImages = new PImage[rows][cols];

  // load images
  for (int r = 0; r < rows; r++) {
    char rowChar = letters.charAt(r);
    for (int c = 0; c < cols; c++) {
      String filename = rowChar + "-" + nf(c + 1, 2) + ".png";
      images[r][c] = loadImage(filename);
      
      // Convert to binary with random halftone
      if (images[r][c] != null) {
        int scaleFactor = 2; // Adjust this value to control how much to scale up
        PImage scaledImg = createImage(images[r][c].width * scaleFactor, images[r][c].height * scaleFactor, RGB);
        scaledImg.copy(images[r][c], 0, 0, images[r][c].width, images[r][c].height, 0, 0, scaledImg.width, scaledImg.height);
        binaryImages[r][c] = createBinaryHalftone(scaledImg);
      }
    }
  }
  
  // text size
  textSize(8);
}

void draw() {
  background(255);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      PImage img = binaryImages[r][c];  // Use the binary version
      if (img != null) {
        // img height and width
        float imgWidth = img.width;
        float imgHeight = img.height;
        // calculate the aspect ratio
        float aspectRatio = imgWidth / imgHeight;
        // calculate the new width and height
        if (aspectRatio > 1) {
          imgWidth = tileWidth;
          imgHeight = tileWidth / aspectRatio;
        } else {
          imgHeight = tileHeight;
          imgWidth = tileHeight * aspectRatio;
        }
        // random 90 degree rotation
        int rotation = int(random(4));
        pushMatrix();
        translate(c * tileWidth + imgWidth/2, r * tileHeight + imgHeight/2);
        // rotate
        // rotate(radians(rotation * 90));
        imageMode(CENTER);
        image(img, 0, 0, imgWidth, imgHeight);
        fill(255, 0, 0);
        text(letters.charAt(r) + "-" + nf(c + 1, 2), 0, - 5);
        popMatrix();
      }
    }
  }
  
  // Save all processed images if 's' is pressed
  if (keyPressed && key == 's') {
    saveProcessedImages();
  }
}

// Function to convert an image to binary using random halftone
PImage createBinaryHalftone(PImage sourceImg) {
  PImage result = createImage(sourceImg.width, sourceImg.height, RGB);
  sourceImg.loadPixels();
  result.loadPixels();
  
  float threshold = 127; // Middle gray threshold

  int margin = 3;
  
  for (int y = 0; y < sourceImg.height; y++) {
    for (int x = 0; x < sourceImg.width; x++) {
      if (x < margin || x >= sourceImg.width - margin || y < margin || y >= sourceImg.height - margin) {
        result.pixels[x + y * sourceImg.width] = color(255); // Set border to white
        continue;
      }

      int loc = x + y * sourceImg.width;
      
      // Get the color
      color pixelColor = sourceImg.pixels[loc];
      
      // Convert to grayscale
      float brightness = brightness(pixelColor);
      
      // Apply random halftone
      float randomOffset = random(-30, 30); // Random variation for the halftone effect
      
      // Set pixel to either black or white based on brightness and random variation
      if (brightness + randomOffset < threshold) {
        result.pixels[loc] = color(0); // Black
      } else {
        result.pixels[loc] = color(255); // White
      }
    }
  }
  
  result.updatePixels();
  return result;
}

// Function to save all processed images
void saveProcessedImages() {
  for (int r = 0; r < rows; r++) {
    char rowChar = letters.charAt(r);
    for (int c = 0; c < cols; c++) {
      if (binaryImages[r][c] != null) {
        String saveFilename = "binary_" + rowChar + "-" + nf(c + 1, 2) + ".png";
        binaryImages[r][c].save(saveFilename);
      }
    }
  }
  println("All images saved!");
}