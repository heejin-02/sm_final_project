package com.smhrd.web.admin;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/admin")
public class AdminController {

    @Autowired
    private AdminService adminService;

    @GetMapping("/insert")
    public String insert() {
        return "insert";
    }

    // 검색+페이징
    @GetMapping("/admin")
    public String farmList(
            @RequestParam(value = "searchField", required = false) String searchField,
            @RequestParam(value = "keyword", required = false) String keyword,
            @RequestParam(value = "page", defaultValue = "1") int page,
            Model model) {

        int pageSize = 10;
        int totalCount;
        List<AdminDTO> farmList;

        if (keyword != null && !keyword.isEmpty() && searchField != null && !searchField.isEmpty()) {
            totalCount = adminService.getSearchCount(searchField, keyword);
            int totalPage = (int) Math.ceil((double) totalCount / pageSize);
            if (page < 1)
                page = 1;
            if (page > totalPage)
                page = totalPage;

            farmList = adminService.searchFarmListWithPaging(searchField, keyword, page, pageSize);
            model.addAttribute("totalPage", totalPage);
        } else {
            totalCount = adminService.getTotalCount();
            int totalPage = (int) Math.ceil((double) totalCount / pageSize);
            if (page < 1)
                page = 1;
            if (page > totalPage)
                page = totalPage;

            farmList = adminService.getFarmListWithPaging(page, pageSize);
            model.addAttribute("totalPage", totalPage);
        }

        model.addAttribute("farmList", farmList);
        model.addAttribute("page", page);
        model.addAttribute("keyword", keyword);
        model.addAttribute("searchField", searchField);

        return "adminPage";
    }

    @PostMapping("/insertUser")
    public String insertUser(AdminDTO dto, Model model) {
        boolean isInserted = adminService.insertUser(dto);
        if (!isInserted) {
            model.addAttribute("msg", "이미 등록되어있는 회원입니다!");
            return "insertForm";
        }
        return "redirect:/admin";
    }

    @PostMapping("/insertFarm")
    public String insertFarm(AdminDTO dto, Model model) {

        MultipartFile file = dto.getFarmImg();
        System.out.println(dto);
        if (file != null && !file.isEmpty()) {
            System.out.println("1");
            String fileName = file.getOriginalFilename();
            String uploadPath = "C:/upload/farm/"; // 폴더 경로설정 다시 하기

            File saveFile = new File(uploadPath + fileName);

            try {
                System.out.println("2");

                file.transferTo(saveFile);
                dto.setFarmImgName(fileName); // DB에 저장할 파일명 세팅
            } catch (IOException e) {
                System.out.println("3");

                e.printStackTrace();
                model.addAttribute("msg", "파일 저장 실패");
                return "insertFarm";
            }
        }

        boolean result = adminService.insertFarm(dto);
        System.out.println("12");

        if (result) {
            System.out.println("13");

            return "redirect:/admin";
        } else {
            model.addAttribute("msg", "등록 실패");
            return "insertFarm";
        }
    }

    // 농장주 상세보기
    @GetMapping("/admin/edit")
    public String viewUserDetail(@RequestParam("userPhone") String userPhone, Model model) {
        AdminDTO user = adminService.getUserByPhone(userPhone);
        model.addAttribute("user", user);
        return "adminUserDetail"; // 상세보기 JSP 파일명
    }

}
