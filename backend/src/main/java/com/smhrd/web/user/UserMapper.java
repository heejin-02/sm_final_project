package com.smhrd.web.user;

import java.util.List;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface UserMapper {
    List<UserDTO> findFarmsByUserPhone(String userPhone);
}
