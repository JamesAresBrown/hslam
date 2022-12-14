#ifndef PMVS3_PATCHORGANIZERS_H
#define PMVS3_PATCHORGANIZERS_H

#include "patch.h"
#include <queue>
// 组织面片
namespace PMVS3 {

class CfindMatch;

class P_compare {
public:
  bool operator()(const Patch::Ppatch& lhs, const Patch::Ppatch& rhs) const {
    return lhs->m_tmp < rhs->m_tmp;
  }
};
 
class CpatchOrganizerS {
 public:
  CpatchOrganizerS(CfindMatch& findMatch);

  void init(void);
  void collectPatches(const int target = 0);
  void collectPatches(std::priority_queue<Patch::Ppatch, std::vector<Patch::Ppatch>,
                      P_compare>& pqpatches);
  
  void collectPatches(const int index,
                      std::priority_queue<Patch::Ppatch, std::vector<Patch::Ppatch>,
                      P_compare>& pqpatches);
  void collectNonFixPatches(const int index, std::vector<Patch::Ppatch>& ppatches);
  
  void writePatches2(const std::string prefix, bool bExportPLY, bool bExportPatch, bool bExportPSet);
  
  void writePLY(const std::vector<Patch::Ppatch>& patches,
                const std::string filename);
  void writePLY(const std::vector<Patch::Ppatch>& patches,
                const std::string filename,
                const std::vector<Vec3i>& colors);
  
  void readPatches(void);
  
  void clearCounts(void);
  void clearFlags(void);

  void setGridsImages(Patch::Cpatch& patch,
                      const std::vector<int>& images) const;
  void addPatch(Patch::Ppatch& ppatch);
  void removePatch(const Patch::Ppatch& ppatch);
  void setGrids(Patch::Ppatch& ppatch) const;
  void setGrids(Patch::Cpatch& patch) const;
  void setVImagesVGrids(Patch::Ppatch& ppatch);
  void setVImagesVGrids(Patch::Cpatch& patch);
  void updateDepthMaps(Patch::Ppatch& ppatch);
  
  int isVisible(const Patch::Cpatch& patch, const int image,
                const int& ix, const int& iy,
                const float strict, const int lock);
  int isVisible0(const Patch::Cpatch& patch, const int image,
                 int& ix, int& iy,
                 const float strict, const int lock);

  void findNeighbors(const Patch::Cpatch& patch,
                     std::vector<Patch::Ppatch>& neighbors,
                     const int lock,
                     const float scale = 1.0f,
                     const int margin = 1,
                     const int skipvis = 0);
  
  void setScales(Patch::Cpatch& patch) const;
  
  float computeUnit(const Patch::Cpatch& patch) const;

  // change the contents of m_images from images to indexes
  // 将 m_images 的内容从图像更改为索引
  void image2index(Patch::Cpatch& patch);
  // change the contents of m_images from indexes to images
  // 将 m_images 的内容从索引更改为图像
  void index2image(Patch::Cpatch& patch);
  
  //----------------------------------------------------------------------
  // Widths of grids 网格的长宽
  std::vector<int> m_gwidths;
  std::vector<int> m_gheights;
  //----------------------------------------------------------------------
  // image, grid 网格化后的图像
  //   1图像索引1维，2网格索引1维，3面片索引1维
  std::vector<std::vector<std::vector<Patch::Ppatch> > > m_pgrids;  
  // image, grid 网格化后的图像
  std::vector<std::vector<std::vector<Patch::Ppatch> > > m_vpgrids;
  // Closest patch  最接近的patch
  std::vector<std::vector<Patch::Ppatch> > m_dpgrids;

  // all the patches in the current level of m_pgrids
  // 当前级别 m_pgrids 中的所有补丁（重要！主要数据结构）
  std::vector<Patch::Ppatch> m_ppatches;

  // Check how many times patch optimization was performed for expansion
  // 检查为扩展执行了多少次补丁优化
  std::vector<std::vector<unsigned char> > m_counts;

  static Patch::Ppatch m_MAXDEPTH;
  static Patch::Ppatch m_BACKGROUND;
  
 protected:
  CfindMatch& m_fm;
};
};

#endif //PMVS3_PATCHORGANIZERS_H
