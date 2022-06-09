#ifndef ETA_MYMODEL_CUH
#define ETA_MYMODEL_CUH

void normalize_vertex_coord(std::vector<num>& vertex_coord) {
    int num_of_vertices{ static_cast<int>(vertex_coord.size()) / 3 };

    num min_x{ ETA_inf };
    num max_x{ -ETA_inf };
    num min_y{ ETA_inf };
    num max_y{ -ETA_inf };
    num min_z{ ETA_inf };
    num max_z{ -ETA_inf };

    for (int i{ 0 }; i < num_of_vertices; ++i) {
        num x{ vertex_coord[3 * i + 0] };
        num y{ vertex_coord[3 * i + 1] };
        num z{ vertex_coord[3 * i + 2] };

        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
        min_z = std::min(min_z, z);
        max_z = std::max(max_z, z);
    }

    num center_x{ (min_x + max_x) / 2 };
    num center_y{ (min_y + max_y) / 2 };
    num center_z{ (min_z + max_z) / 2 };

    num scale_x{ 2 / (max_x - min_x) };
    num scale_y{ 2 / (max_y - min_y) };
    num scale_z{ 2 / (max_z - min_z) };

    num scale{ std::min(scale_x, std::min(scale_y, scale_z)) };

    for (int i{ 0 }; i < num_of_vertices; ++i) {
        num& x{ vertex_coord[3 * i + 0] };
        num& y{ vertex_coord[3 * i + 1] };
        num& z{ vertex_coord[3 * i + 2] };

        x = (x - center_x) * scale;
        y = (y - center_y) * scale;
        z = (z - center_z) * scale;
    }
}

struct MyModel {
    int num_of_vertices;
    int num_of_faces;

    std::vector<int> face_id;
    std::vector<num> vertex_coord;
    std::vector<num> vertex_normal;
    std::vector<num> vertex_color;
    std::vector<num> vertex_texture_coord;

    VectorView face_id_gpu;
    VectorView face_coord_gpu;
    VectorView vertex_normal_view;
    VectorView vertex_color_view;
    VectorView vertex_texture_coord_view;

    Image texture_img;
    Texture2D<Vector<num, 3>> texture;

    Matrix<num, 4, 4> transform;

    bool has_texture;

    bool loaded{ false };

    void Release() {
        if (!this->loaded) { return; }
        //
    }

    void LoadModel(const std::string& path, bool load_texture = false) {
        this->Release();

        this->vertex_coord =
            read_vector<num>(join_path(path, "face_coord.txt"));
        normalize_vertex_coord(vertex_coord);

        this->vertex_normal =
            read_vector<num>(join_path(path, "face_normal.txt"));

        this->vertex_color =
            read_vector<num>(join_path(path, "face_color.txt"));

        this->num_of_vertices = static_cast<int>(this->vertex_coord.size()) / 3;
        this->num_of_faces = this->num_of_vertices / 3;

        this->transform = Matrix<num, 4, 4>::eye();

        this->face_id.resize(this->num_of_faces);

        this->face_id_gpu.base = gpu_malloc(sizeof(int) * this->num_of_faces);
        this->face_id_gpu.coeff[0] = sizeof(int);

        this->face_coord_gpu.base =
            gpu_malloc(sizeof(Vector<num, 3>) * this->num_of_vertices);
        this->face_coord_gpu.coeff[0] = sizeof(Vector<num, 3>) * 3;
        this->face_coord_gpu.coeff[1] = sizeof(Vector<num, 3>);

        this->vertex_color_view.base = this->vertex_color.data();
        this->vertex_color_view.coeff[0] = sizeof(num) * 3 * 3;
        this->vertex_color_view.coeff[1] = sizeof(num) * 3;
        this->vertex_color_view.coeff[2] = sizeof(num);

        this->has_texture = load_texture;

        if (this->has_texture) {
            this->vertex_texture_coord =
                read_vector<num>(join_path(path, "face_color_tex_coord.txt"));

            this->vertex_texture_coord_view.base =
                this->vertex_texture_coord.data();
            this->vertex_texture_coord_view.coeff[0] =
                sizeof(Vector<num, 2>) * 3;
            this->vertex_texture_coord_view.coeff[1] = sizeof(Vector<num, 2>);

            this->texture_img.load(join_path(path, "face_color_tex.png"));

            this->texture.height = this->texture_img.height();
            this->texture.width = this->texture_img.width();

            this->texture.s_wrapping_mode =
                Texture2DWrappingMode::MIRRORED_REPEAT;
            this->texture.t_wrapping_mode =
                Texture2DWrappingMode::MIRRORED_REPEAT;

            this->texture.data.base =
                cpu_malloc(sizeof(num) * 3 * this->texture_img.size());
            this->texture.data.coeff[0] =
                sizeof(num) * 3 * this->texture_img.width();
            this->texture.data.coeff[1] = sizeof(num) * 3;

            for (int i{ 0 }; i < this->texture_img.height(); ++i) {
                for (int j{ 0 }; j < this->texture_img.width(); ++j) {
                    Vector<num, 3>& v{
                        this->texture.data.get_ref<Vector<num, 3>>(i, j)
                    };

                    v.data[0] = this->texture_img.get(i, j, 0) / 255.0f;
                    v.data[1] = this->texture_img.get(i, j, 1) / 255.0f;
                    v.data[2] = this->texture_img.get(i, j, 2) / 255.0f;
                }
            }
        }
    }

    void LoadToGpu(cudaStream_t stream = 0) {
        cudaMemcpyAsync(this->face_id_gpu.base, this->face_id.data(),
                        sizeof(int) * this->num_of_faces,
                        cudaMemcpyHostToDevice, stream);

        cudaMemcpyAsync(this->face_coord_gpu.base, this->vertex_coord.data(),
                        sizeof(Vector<num, 3>) * this->num_of_vertices,
                        cudaMemcpyHostToDevice, stream);
    }
};

#endif