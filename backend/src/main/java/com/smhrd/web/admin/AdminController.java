package com.smhrd.web.admin;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import io.swagger.v3.oas.annotations.Operation;

@RestController
@RequestMapping("/admin")
public class AdminController {

    @Autowired
    private AdminService adminService;

    // 회원 + 농장 리스트 (검색 + 페이징)
    @Operation(summary = "회원+농장 리스트 조회", description = "검색 및 페이징 포함")
    @GetMapping("/farms")
    public ResponseEntity<?> farmList(
            @RequestParam(value = "searchField", required = false) String searchField,
            @RequestParam(value = "keyword", required = false) String keyword,
            @RequestParam(value = "page", defaultValue = "1") int page) {

        int pageSize = 10;
        int totalCount;
        List<AdminDTO> farmList;

        if (keyword != null && !keyword.isEmpty() && searchField != null && !searchField.isEmpty()) {
            totalCount = adminService.getSearchCount(searchField, keyword);
            int totalPage = (int) Math.ceil((double) totalCount / pageSize);
            if (page < 1) page = 1;
            if (page > totalPage) page = totalPage;

            farmList = adminService.searchFarmListWithPaging(searchField, keyword, page, pageSize);
            return ResponseEntity.ok(new PagingResult<>(farmList, page, totalPage));
        } else {
            totalCount = adminService.getTotalCount();
            int totalPage = (int) Math.ceil((double) totalCount / pageSize);
            if (page < 1) page = 1;
            if (page > totalPage) page = totalPage;

            farmList = adminService.getFarmListWithPaging(page, pageSize);
            return ResponseEntity.ok(new PagingResult<>(farmList, page, totalPage));
        }
    }

    // 회원 등록 (JSON x, form-data + multipartFile)
    @Operation(summary = "회원 등록")
    @PostMapping("/insertUser")
    public ResponseEntity<?> insertUser(@RequestBody AdminDTO dto) {
        boolean isInserted = adminService.insertUser(dto);
        if (!isInserted) {
            return ResponseEntity.badRequest().body("이미 등록된 회원입니다.");
        }
        return ResponseEntity.ok("회원 등록 성공");
    }

    // 농장 등록 (multipart/form-data + 파일업로드)
    @Operation(summary = "농장 등록")
    @PostMapping(value = "/insertFarm", consumes = {"multipart/form-data"})
    public ResponseEntity<?> insertFarm(
            @RequestPart("farmImg") MultipartFile file,
            @RequestPart("dto") AdminDTO dto) {

        if (file != null && !file.isEmpty()) {
            String fileName = file.getOriginalFilename();
            String uploadPath = "C:/upload/farm/";

            File saveFile = new File(uploadPath + fileName);
            try {
                file.transferTo(saveFile);
                dto.setFarmImgName(fileName);
            } catch (IOException e) {
                e.printStackTrace();
                return ResponseEntity.internalServerError().body("파일 저장 실패");
            }
        }

        boolean result = adminService.insertFarm(dto);
        if (result) {
            return ResponseEntity.ok("농장 등록 성공");
        } else {
            return ResponseEntity.badRequest().body("농장 등록 실패");
        }
    }

    // 회원 상세조회 (userPhone 파라미터로 JSON 반환)
    @Operation(summary = "회원 상세조회")
    @GetMapping("/user/{userPhone}")
    public ResponseEntity<AdminDTO> viewUserDetail(@PathVariable String userPhone) {
        AdminDTO user = adminService.getUserByPhone(userPhone);
        if (user == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(user);
    }

    // 페이징 응답 DTO (내부 클래스 or 별도 파일로 분리 가능)
    public static class PagingResult<T> {
        private List<T> data;
        private int page;
        private int totalPage;

        public PagingResult(List<T> data, int page, int totalPage) {
            this.data = data;
            this.page = page;
            this.totalPage = totalPage;
        }

        // getters & setters
        public List<T> getData() { return data; }
        public void setData(List<T> data) { this.data = data; }
        public int getPage() { return page; }
        public void setPage(int page) { this.page = page; }
        public int getTotalPage() { return totalPage; }
        public void setTotalPage(int totalPage) { this.totalPage = totalPage; }
    }
}
