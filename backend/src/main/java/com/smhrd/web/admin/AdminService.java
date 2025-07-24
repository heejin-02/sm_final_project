package com.smhrd.web.admin;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class AdminService {

    @Autowired
    private AdminMapper adminMapper;

    public List<AdminDTO> getFarmList() {
        return adminMapper.selectFarmList();
    }
    
    public List<AdminDTO> searchFarmList(String keyword) {
        return adminMapper.searchFarmList(keyword);
    }

    
    public boolean insertUser(AdminDTO user) {
        AdminDTO existingUser = adminMapper.selectUserByPhone(user.getUserPhone());
        if (existingUser != null) {
            return false;
        }
        adminMapper.insertUser(user);
        return true;
    }
    
    public boolean insertFarm(AdminDTO dto) {
        return adminMapper.insertFarm(dto) > 0;
    }

    // 전체 글 개수 (검색 없을 때)
    public int getTotalCount() {
        return adminMapper.selectTotalCount();
    }

    // 검색 결과 글 개수
    public int getSearchCount(String keyword) {
        return adminMapper.selectSearchCount(keyword);
    }

    // 페이징 처리를 위해 시작 행(startRow)과 끝 행(endRow), 
	public List<AdminDTO> getFarmListWithPaging(int page, int pageSize) {
	    int startRow = (page -1) * pageSize + 1;
	    int endRow = page * pageSize;
	    Map<String, Object> params = new HashMap<>();
	    params.put("startRow", startRow);
	    params.put("endRow", endRow);
	    return adminMapper.selectFarmListWithPaging(params);
	}
	
	public List<AdminDTO> searchFarmListWithPaging(String keyword, int page, int pageSize) {
	    int startRow = (page -1) * pageSize + 1;
	    int endRow = page * pageSize;
	    Map<String, Object> params = new HashMap<>();
	    params.put("keyword", keyword);
	    params.put("startRow", startRow);
	    params.put("endRow", endRow);
	    return adminMapper.searchFarmListWithPaging(params);
	}
	
	// 검색 결과 글 개수
	public int getSearchCount(String searchField, String keyword) {
	    return adminMapper.selectSearchCount(searchField, keyword);
	}

	// 검색 있을 때 페이징 리스트
	public List<AdminDTO> searchFarmListWithPaging(String searchField, String keyword, int page, int pageSize) {
	    int startRow = (page -1) * pageSize + 1;
	    int endRow = page * pageSize;
	    return adminMapper.searchFarmListWithPaging(searchField, keyword, startRow, endRow);
	}


	public AdminDTO getUserByPhone(String userPhone) {
	    return adminMapper.getUserByPhone(userPhone);
	}

}
