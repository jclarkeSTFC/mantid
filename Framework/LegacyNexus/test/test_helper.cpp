#include "test_helper.h"
#include <cstdarg>
#include <filesystem>

/**
 * Let's face it, std::string is poorly designed,
 * and this is the constructor that it needed to have.
 * Initialize a string from a c-style formatting string.
 */
std::string LegacyNexusTest::strmakef(const char *const fmt, ...) {
  char buf[256];

  va_list args;
  va_start(args, fmt);
  const auto r = std::vsnprintf(buf, sizeof buf, fmt, args);
  va_end(args);

  if (r < 0)
    // conversion failed
    return {};

  const size_t len = r;
  if (len < sizeof buf)
    // we fit in the buffer
    return {buf, len};

  std::string s(len, '\0');
  va_start(args, fmt);
  std::vsnprintf(&(*s.begin()), len + 1, fmt, args);
  va_end(args);
  return s;
}

LegacyNexusTest::FormatUniqueVars LegacyNexusTest::getFormatUniqueVars(const LegacyNexusTest::NexusFormat fmt,
                                                                       const std::string &filename) {
  std::string relFilePath;
  std::string fileExt;
  const std::string extSubStr = (filename.size() > 4) ? filename.substr(filename.size() - 4) : "";
  switch (fmt) {
  case LegacyNexusTest::NexusFormat::HDF4:
    fileExt = (extSubStr == ".nxs") ? "" : ".h4";
    relFilePath = "LegacyNexus/hdf4/" + filename + fileExt;
    break;
  case LegacyNexusTest::NexusFormat::HDF5:
    fileExt = (extSubStr == ".nxs") ? "" : ".h5";
    relFilePath = "LegacyNexus/hdf5/" + filename + fileExt;
    break;
  }
  return LegacyNexusTest::FormatUniqueVars{relFilePath};
}
