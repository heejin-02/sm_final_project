package com.smhrd.web.admin;

import java.util.List;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface AdminMapper {
    List<AdminDTO> selectAllUsers();
    List<AdminDTO> selectUserByPhone(String userPhone);

    int updateFarmInfo(AdminDTO farm);
    int deleteFarm(Long farmIdx);
    int updateUserInfo(AdminDTO user);
    int deleteUser(String userPhone);


}
