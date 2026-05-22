#include <astra/astra.hpp>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>

int main() {
    astra::initialize();

    astra::StreamSet streamSet;
    astra::StreamReader reader = streamSet.create_reader();

    auto colorStream = reader.stream<astra::ColorStream>();
    colorStream.start();

    std::cout << "Waiting for color frame..." << std::endl;

    for (int i = 0; i < 100; ++i) {
        astra_update();

        auto frame = reader.get_latest_frame();
        auto colorFrame = frame.get<astra::ColorFrame>();

        if (colorFrame.is_valid()) {
            int width = colorFrame.width();
            int height = colorFrame.height();

            std::cout << "Color frame: " << width << " x " << height << std::endl;

            const astra::RgbPixel* data = colorFrame.data();

            std::ofstream out("astra_color.ppm", std::ios::binary);
            out << "P6\n" << width << " " << height << "\n255\n";

            for (int p = 0; p < width * height; ++p) {
                out.put(data[p].r);
                out.put(data[p].g);
                out.put(data[p].b);
            }

            out.close();

            std::cout << "Saved: astra_color.ppm" << std::endl;

            colorStream.stop();
            astra::terminate();
            return 0;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    std::cerr << "Failed to get color frame" << std::endl;

    colorStream.stop();
    astra::terminate();
    return 1;
}