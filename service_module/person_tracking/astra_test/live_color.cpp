#include <astra/astra.hpp>
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    astra::initialize();

    astra::StreamSet streamSet;
    astra::StreamReader reader = streamSet.create_reader();

    auto colorStream = reader.stream<astra::ColorStream>();
    colorStream.start();

    std::cout << "Starting live color frame test..." << std::endl;

    int frame_count = 0;
    auto start = std::chrono::steady_clock::now();

    while (true) {
        astra_update();

        auto frame = reader.get_latest_frame();
        auto colorFrame = frame.get<astra::ColorFrame>();

        if (colorFrame.is_valid()) {
            frame_count++;

            int width = colorFrame.width();
            int height = colorFrame.height();

            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();

            if (elapsed >= 1.0) {
                std::cout << "Color frame: "
                          << width << "x" << height
                          << " | FPS: " << frame_count / elapsed
                          << std::endl;

                frame_count = 0;
                start = now;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    colorStream.stop();
    astra::terminate();
    return 0;
}