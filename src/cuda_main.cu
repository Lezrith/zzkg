#include <stdio.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <fstream>
#include <limits>
#include <cuda.h>
#include <algorithm>
#include "check_cuda_error.h"

#define BLOCK_DIM 1024
#define GRID_DIM 4

constexpr int warp_size = 4;
constexpr int chunk_size = 8;

struct graph {
    graph(int rows, int cols) : vertices(rows * cols),
                                bfs(rows * cols, std::numeric_limits<int>::max()),
                                colors(rows * cols, 0),
                                cols(cols) {}

    const unsigned long number_of_vertices() const {
        return vertices.size() - 1;
    }

    const unsigned long number_of_edges() const {
        return edges.size();
    }

    std::vector<int> vertices;
    std::vector<int> edges;
    std::vector<int> bfs;
    std::vector<int> colors;
    int cols;
};

std::vector<std::pair<int, int>> get_neighbour_indices(int x, int y) {
    return {
            {x - 1, y - 1},
            {x,     y - 1},
            {x + 1, y - 1},
            {x + 1, y},
            {x + 1, y + 1},
            {x,     y + 1},
            {x - 1, y + 1},
            {x - 1, y},
    };
}

bool is_diagonal(int x, int y, int x2, int y2) {
    return abs(x - x2) == 1 && abs(y - y2) == 1;
}

bool is_map_edge(const std::vector<std::vector<int>> &map, int x, int y) {
    return x < 0 || x >= map[0].size() || y < 0 || y >= map.size();
}

bool is_neighbour(const std::vector<std::vector<int>> &map, int x, int y, int x2, int y2) {
    if (is_map_edge(map, x, y) || is_map_edge(map, x2, y2)) {
        return false;
    }

    int value = map[y][x];
    int value2 = map[y2][x2];

    return value == value2 && (value > 0 || !is_diagonal(x, y, x2, y2));
}

int to_index(int x, int y, int cols) {
    return y * cols + x;
}

bool is_shallow(const std::vector<std::vector<int>> &map, int x, int y) {
    auto neighbours = get_neighbour_indices(x, y);
    return map[y][x] == 0 && std::any_of(neighbours.begin(), neighbours.end(), [map](const std::pair<int, int> &p) {
        return !is_map_edge(map, p.first, p.second) && map[p.second][p.first] == 1;
    });
}

bool draw_route(graph &graph, const std::vector<std::vector<int>> &map) {
    int x = graph.cols - 1, y = -1, current = 0;
    for (int i = 0; i < map.size(); ++i) {
        if (map[i][x] == 0
            && (y == -1 || graph.bfs[to_index(x, i, graph.cols)] > graph.bfs[to_index(x, y, graph.cols)])
            && graph.bfs[to_index(x, i, graph.cols)] != std::numeric_limits<int>::max()) {
            y = i;
            current = graph.bfs[to_index(x, i, graph.cols)];
        }
    }
    if (y == -1) {
        return false;
    } else {
        graph.colors[to_index(x, y, graph.cols)] = 1;
        while (x != 0) {
            auto neighbours = get_neighbour_indices(x, y);
            for (const auto &p : neighbours) {
                int index = to_index(x, y, graph.cols);
                auto neighbour = std::find_if(neighbours.begin(), neighbours.end(), [&](const auto &p) {
                    return is_neighbour(map, x, y, p.first, p.second) &&
                           graph.bfs[to_index(p.first, p.second, graph.cols)] == current - 1 &&
                           graph.colors[index] != -1;
                });
                if (neighbour != neighbours.end()) {
                    graph.colors[index] = 1;
                    y = neighbour->second;
                    x = neighbour->first;
                    current--;
                    break;
                } else {
                    return false;
                }
            }
        }
        graph.colors[to_index(x, y, graph.cols)] = 1;
    }
    return true;
}

std::pair<graph, std::vector<std::vector<int>>> read_file(const std::string &filename) {
    std::ifstream file;

    file.open(filename, std::ios::in);

    if (file.good()) {
        int rows = 0;
        int cols = 0;

        file >> rows >> cols;
        graph graph(rows, cols);

        std::vector<std::vector<int>> map(rows, std::vector<int>(cols));

        for (size_t i = 0; i < rows; i++) {
            std::vector<int> &row = map[i];
            for (size_t j = 0; j < cols; j++) {
                file >> row[j];
            }
        }

        for (size_t y = 0; y < rows; y++) {
            for (size_t x = 0; x < cols; x++) {
                int index = to_index(x, y, cols);
                graph.vertices[index] = graph.edges.size();
                if (is_shallow(map, x, y)) {
                    graph.colors[index] = -1;
                } else {
                    auto neighbours = get_neighbour_indices(x, y);
                    for (auto p : neighbours) {
                        if (is_neighbour(map, x, y, p.first, p.second)) {
                            graph.edges.emplace_back(to_index(p.first, p.second, cols));
                        }
                    }
                    graph.colors[index] = map[y][x];
                }
            }
        }
        graph.vertices.push_back(graph.edges.size());
        file.close();
        return {graph, map};
    } else {
        throw std::runtime_error("bad file");
    }
}

void save_output(const std::string &filename, const graph &graph) {
    std::ofstream file;

    file.open(filename, std::ios::out);

    if (file.good()) {
        for (int i = 0; i < graph.colors.size(); ++i) {
            file << (graph.colors[i] >= 0 ? graph.colors[i] : 0);
            if ((i + 1) % graph.cols == 0) {
                file << std::endl;
            } else {
                file << " ";
            }
        }
        file.close();
    } else {
        throw std::runtime_error("bad file");
    }

    file.open("routes.txt", std::ios::out);

    if (file.good()) {
        for (int i = 0; i < graph.bfs.size(); ++i) {
            file << graph.bfs[i];
            if ((i + 1) % graph.cols == 0) {
                file << std::endl;
            } else {
                file << " ";
            }
        }
        file.close();
    } else {
        throw std::runtime_error("bad file");
    }
}


//  WARPSIZE – liczba wątków w warpie
//  CHUNKSIZE – liczba węzłów przetwarzanych przez warp
//  INFINITY – liczba określająca wartość BFS węzłów jeszcze nie odwiedzonych
template<int WARPSIZE, int CHUNKSIZE, int INF>
__global__ void
bfs_kernel(const int *V, int *E, int *BFS, int *C, int color, unsigned long N, int curr, bool *finished) {
    extern __shared__ int base[];
    int t = blockIdx.x * blockDim.x + threadIdx.x; //globalny nr wątku
    int lane = threadIdx.x % WARPSIZE;             //nr wątku w warpie
    int warpLocal = threadIdx.x / WARPSIZE;        //nr warpa w bloku
    int warpGlobal = t / WARPSIZE;                 //globalny nr warpa
    int warpLocalCount = blockDim.x / WARPSIZE;    //liczba warpow w bloku

    // Równoległe przepisanie części danych do pamięci współdzielonej
    // (pętla sekwencyjno-równoległa)
    int *sV = base + warpLocal * (CHUNKSIZE + 1);
    int *sBFS = base + warpLocalCount * (CHUNKSIZE + 1) + warpLocal * CHUNKSIZE;
    for (int i = lane; i < CHUNKSIZE + 1; i += WARPSIZE) { //Przepisz jedną wartość więcej
        if (warpGlobal * CHUNKSIZE + i < N + 1) {
            sV[i] = V[warpGlobal * CHUNKSIZE + i];
        }
    }
    for (int i = lane; i < CHUNKSIZE; i += WARPSIZE) {
        if (warpGlobal * CHUNKSIZE + i < N) {
            sBFS[i] = BFS[warpGlobal * CHUNKSIZE + i];
        }
    }

    __threadfence_block(); //Wszystkie wątki powinny widzieć odczytane dane
    for (int v = 0; v < CHUNKSIZE; v++) {    //Przeglądaj kolejne węzły w zbiorze warpa
        if (v + warpGlobal * CHUNKSIZE < N) {  //Jeżeli nie wychodzimy poza tablicę
            if (sBFS[v] == curr) {   //Jeżeli do węzła dotarto w poprzedniej iteracji
                // Iteruj po sąsiadach
                int num_nbr = sV[v + 1] - sV[v]; //Liczba sąsiadów (po to 1 element więcej)
                int *nbrs = &E[sV[v]];           //Wskaźnik na listę sąsiadów
                // ”Równoległo–sekwencyjna” pętla przeglądająca sąsiadów węzła v
                for (int i = lane; i < num_nbr; i += WARPSIZE) {
                    int w = nbrs[i];             //Numer sąsiada
                    if (BFS[w] == INF && C[w] != -1) {         //Jeżeli sąsiada jeszcze nie odwiedzono
                        *finished = false;       //Konieczna ponowna iteracja
                        BFS[w] = curr + 1;       //Zapisz numer BFS sąsiada
                        if (C[w] != 0) {
                            C[w] = color;
                        }
                    }
                }
                __threadfence_block();       //Wszystkie wątki powinny zobaczyć zapisy
            }
        }
    }
}

void bfs(graph &graph, int start, int color) {
    int *V, *E, *BFS, *C;
    bool finished = true, *finished_device;
    unsigned long number_of_vertices = graph.number_of_vertices();
    unsigned long number_of_edges = graph.number_of_edges();

    graph.bfs[start] = 0;
    if (graph.colors[start] != 0) {
        graph.colors[start] = color;
    }

    CudaSafeCall(cudaMalloc((void **) &V, (number_of_vertices + 1) * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **) &E, number_of_edges * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **) &BFS, number_of_vertices * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **) &C, number_of_vertices * sizeof(int)));
    CudaSafeCall(cudaMalloc((void **) &finished_device, sizeof(bool)));

    CudaSafeCall(cudaMemcpy(V, graph.vertices.data(), (number_of_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(E, graph.edges.data(), number_of_edges * sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(BFS, graph.bfs.data(), number_of_vertices * sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(C, graph.colors.data(), number_of_vertices * sizeof(int), cudaMemcpyHostToDevice));

    int shared_mem_size =
            2 * BLOCK_DIM / warp_size * (chunk_size + 1) * sizeof(int);
    int curr = 0;
    do {
        finished = true;
        CudaSafeCall(cudaMemcpy(finished_device, &finished, sizeof(bool), cudaMemcpyHostToDevice));
        bfs_kernel<warp_size, chunk_size, std::numeric_limits<int>::max()>
                << < GRID_DIM, BLOCK_DIM, shared_mem_size >> >
                                          (V, E, BFS, C, color, number_of_vertices, curr, finished_device);
        CudaCheckError();
        CudaSafeCall(cudaMemcpy(&finished, finished_device, sizeof(bool), cudaMemcpyDeviceToHost));
        curr++;
    } while (!finished && curr <= number_of_vertices);
    CudaSafeCall(cudaMemcpy(graph.bfs.data(), BFS, number_of_vertices * sizeof(int), cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(graph.colors.data(), C, number_of_vertices * sizeof(int), cudaMemcpyDeviceToHost));
}

int cuda_main(int argc, char **argv) {
    if (argc < 2) {
        return 1;
    } else {
        auto x = read_file(argv[1]);
        auto graph = x.first;
        auto map = x.second;
        int color = 2;
        for (int i = 0; i < graph.number_of_vertices(); ++i) {
            if (graph.bfs[i] == std::numeric_limits<int>::max()
                && (graph.colors[i] != 0 || i % graph.cols == 0)
                && graph.colors[i] != -1) {
                bfs(graph, i, color);
                color++;
            }
        }
        if (draw_route(graph, map)) {
            save_output("out.txt", graph);
        } else {
            std::cout << "Nie ma ścieżki" << std::endl;
        }
    }

    return 0;
}
