package com.smhrd.web.Home;

import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface HomeMapper {
    HomeDTO login(HomeDTO loginUser); 
    
}
