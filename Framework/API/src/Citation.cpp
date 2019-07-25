// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2019 ISIS Rutherford Appleton Laboratory UKRI,
//     NScD Oak Ridge National Laboratory, European Spallation Source
//     & Institut Laue - Langevin
// SPDX - License - Identifier: GPL - 3.0 +

#include "MantidAPI/Citation.h"
#include "MantidAPI/CitationConstructorHelpers.h"

#include <nexus/NeXusFile.hpp>

namespace Mantid {
namespace API {

/**
  This function is designed for construction and validation for loading, however
  will inherently work with getCitation.

  - description is always optional (this isn't needed for citation, but gives
  insight as to why this citation is relevant)
  - if bibtex is provided endnote must also be provided, and vice-versa (BibTex
  and Endnote contain essentially the same information, they can both be created
  if one can be. BibTex and Endnote do not imply a DOI is minted)
   - if doi is provided, url, bibtex and endnote must all be provided (BibTex
  and Endnote can be generated from DOIs)
    - if none of doi, bibtex or endnote are provided, url must be provided
  (there must be something there, even if this isn't citable a URL is better
  than nothing)
 */

Citation::Citation(const std::string &doi, const std::string &bibtex,
                   const std::string &endnote, const std::string &url,
                   const std::string &description) {
  if (doi == "" && bibtex == "" && endnote == "" && url == "" &&
      description == "")
    throw std::invalid_argument("No arguements were given!");

  // This is an initial implementation that expects it but it should be possible
  // to generate one from the other
  if ((bibtex != "" && endnote == "") || (bibtex == "" && endnote != ""))
    throw std::invalid_argument(
        "If bibtex is provided, endnote must also be provided and vice-versa");

  if (doi != "" && (bibtex == "" || endnote == "" || url == ""))
    throw std::invalid_argument(
        "If doi is provided then url, bibtex and endnote must be");

  if (doi == "" && bibtex == "" && endnote == "")
    if (url == "")
      throw std::invalid_argument(
          "If none of doi, bibtex, or endnote is provided, then url must be");

  m_doi = doi;
  m_bibtex = bibtex;
  m_endnote = endnote;
  m_url = url;
  m_description = description;
}

Citation::Citation(::NeXus::File *file, const std::string &group) {
  loadNexus(file, group);
}

Citation::Citation(const BaseCitation &cite)
    : Citation(cite.m_doi, cite.toBibTex(), cite.toEndNote(), cite.m_url,
               cite.m_description) {}

bool Citation::operator==(const Citation &rhs) const {
  return m_bibtex == rhs.m_bibtex && m_description == rhs.m_description &&
         m_doi == rhs.m_doi && m_endnote == rhs.m_endnote && m_url == rhs.m_url;
}

const std::string &Citation::description() const { return m_description; }
const std::string &Citation::url() const { return m_url; }
const std::string &Citation::doi() const { return m_doi; }
const std::string &Citation::bibtex() const { return m_bibtex; }
const std::string &Citation::endnote() const { return m_endnote; }

void Citation::loadNexus(::NeXus::File *file, const std::string &group) {
  file->openGroup(group, "NXCite");
  file->readData("url", m_url);
  file->readData("description", m_description);
  file->readData("doi", m_doi);
  file->readData("endnote", m_endnote);
  // Since it remove the whitespace and it intends to restore it's state
  m_endnote += " \n";
  file->readData("bibtex", m_bibtex);
  file->closeGroup();
}

void Citation::saveNexus(::NeXus::File *file, const std::string &group) {
  file->makeGroup(group, "NXCite", true);
  file->writeData("url", m_url);
  file->writeData("description", m_description);
  file->writeData("doi", m_doi);
  file->writeData("endnote", m_endnote);
  file->writeData("bibtex", m_bibtex);
  file->closeGroup();
}

} // namespace API
} // namespace Mantid