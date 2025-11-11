#include <string>
// void run(std::string config_str, std::string fname, int header_sel);
void run(std::string config_str, std::string output_dir, int header_sel);
void run_interactive_sel(std::string config_str, std::string fname, int header_sel);
void run_interactive_in_place(float *input_array,
			      std::vector<std::string> &fnames,
			      int x, int y, int z, int count, int mode,
			      std::string path_to_bgmap);
