package com.smhrd.web.admin;

import java.util.List;
import java.util.Map;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface AdminMapper {
    List<AdminDTO> selectFarmList();
    
    List<AdminDTO> searchFarmList(String keyword);
    
    public void insertUser(AdminDTO dto);
    AdminDTO selectUserByPhone(String userPhone);

    int insertFarm(AdminDTO dto);
    
    int selectTotalCount();

    int selectSearchCount(@Param("keyword") String keyword);

    List<AdminDTO> selectFarmListWithPaging(Map<String, Object> params);

    List<AdminDTO> searchFarmListWithPaging(Map<String, Object> params);

    
    int selectSearchCount(@Param("searchField") String searchField, @Param("keyword") String keyword);

    List<AdminDTO> searchFarmListWithPaging(
        @Param("searchField") String searchField,
        @Param("keyword") String keyword,
        @Param("startRow") int startRow,
        @Param("endRow") int endRow);

    public AdminDTO getUserByPhone(String userPhone);


}
