package com.smhrd.web.user;

import java.util.List;

import org.apache.ibatis.annotations.Mapper;


@Mapper
public interface UserMapper {

    List<UserDTO> selectFarmsByUserPhone(String userPhone);
    UserDTO selectFarmByFarmIdx(Long farmIdx);
    


}
